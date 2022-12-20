# Copyright (c) SI-Analytics. All rights reserved.
import os
import tempfile
import time
from typing import Optional

import mmcv
import ray.tune as tune
import torch
from ray.air import Checkpoint, session
from torch.optim import Optimizer

from siatune.mm.core import HOOKS, MMENGINE_BASED, BaseRunner
from siatune.mm.core import CheckpointHook as _CheckpointHook
from siatune.mm.core import (get_state_dict, is_module_wrapper, master_only,
                             weights_to_cpu)

if MMENGINE_BASED:

    @HOOKS.register_module()
    class RayCheckpointHook(_CheckpointHook):
        """Save checkpoints periodically."""

        def before_train(self, runner) -> None:
            """This hook omits the setting process because it gets information
            from the ray session.

            Args:
                runner (:obj:`mmcv.runner.BaseRunner`):
                    The runner.
            """
            pass

        def _save_checkpoint(self, runner: BaseRunner) -> None:
            """Save checkpoints periodically.

            Args:
                runner (:obj:`mmcv.runner.BaseRunner`):
                    The runner to save checkpoints.
            """
            import logging

            import mmengine
            from mmengine.dist import is_main_process
            from mmengine.logging import print_log
            from mmengine.optim import OptimWrapper
            from mmengine.utils import get_git_hash

            meta = {}
            if self.by_epoch:
                # self.epoch increments 1 after
                # `self.call_hook('after_train_epoch)` but `save_checkpoint` is
                # called by `after_train_epoch`` method of `CheckpointHook` so
                # `epoch` should be `self.epoch + 1`
                meta.update(epoch=runner.epoch + 1, iter=runner.iter)
            else:
                meta.update(epoch=runner.epoch, iter=runner.iter + 1)

            meta.update(
                cfg=runner.cfg.pretty_text,
                seed=runner.seed,
                experiment_name=runner.experiment_name,
                time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
                mmengine_version=mmengine.__version__ + get_git_hash())

            if hasattr(runner.train_dataloader.dataset, 'metainfo'):
                meta.update(
                    dataset_meta=runner.train_dataloader.dataset.metainfo)

            if is_module_wrapper(runner.model):
                model = runner.model.module
            else:
                model = runner.model

            checkpoint = {
                'meta': meta,
                'state_dict': weights_to_cpu(get_state_dict(model)),
                'message_hub': runner.message_hub.state_dict()
            }
            # save optimizer state dict to checkpoint
            if self.save_optimizer:
                if isinstance(runner.optim_wrapper, OptimWrapper):
                    checkpoint['optimizer'] = runner.optim_wrapper.state_dict()
                else:
                    raise TypeError(
                        'runner.optim_wrapper should be an `OptimWrapper` '
                        'or `OptimWrapperDict` instance, but got '
                        f'{runner.optim_wrapper}')

            # save param scheduler state dict
            if self.save_param_scheduler and runner.param_schedulers is None:
                print_log(
                    '`save_param_scheduler` is True but `runner.param_schedulers` '
                    'is None, so skip saving parameter schedulers',
                    logger='current',
                    level=logging.WARNING)
                self.save_param_scheduler = False
            if self.save_param_scheduler:
                if isinstance(runner.param_schedulers, dict):
                    checkpoint['param_schedulers'] = dict()
                    for name, schedulers in runner.param_schedulers.items():
                        checkpoint['param_schedulers'][name] = []
                        for scheduler in schedulers:
                            state_dict = scheduler.state_dict()
                            checkpoint['param_schedulers'][name].append(
                                state_dict)
                else:
                    checkpoint['param_schedulers'] = []
                    for scheduler in runner.param_schedulers:  # type: ignore
                        state_dict = scheduler.state_dict()  # type: ignore
                        checkpoint['param_schedulers'].append(state_dict)

            step = (runner.epoch + 1) // self.interval
            if not self.by_epoch:
                step //= runner.iter + 1

            with tune.checkpoint_dir(step=step) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'ray_ckpt.pth')
                torch.save(checkpoint, path)

else:

    @HOOKS.register_module()
    class RayCheckpointHook(_CheckpointHook):

        def __init__(self,
                     interval: int = -1,
                     by_epoch: bool = True,
                     save_optimizer: bool = True,
                     max_keep_ckpts: int = -1,
                     save_last: bool = True,
                     sync_buffer: Optional[bool] = False,
                     file_client_args: Optional[dict] = None,
                     **kwargs):
            """Initialize the CheckpointHook.

            Args:
                interval (int): The saving period. If ``by_epoch=True``, interval
                    indicates epochs, otherwise it indicates iterations.
                    Default: -1, which means "never".
                by_epoch (bool): Saving checkpoints by epoch or by iteration.
                    Default: True.
                save_optimizer (bool): Whether to save optimizer state_dict in the
                    checkpoint. It is usually used for resuming experiments.
                    Default: True.
                max_keep_ckpts (int, optional): The maximum checkpoints to keep.
                    In some cases we want only the latest few checkpoints and would
                    like to delete old ones to save the disk space.
                    Default: -1, which means unlimited.
                save_last (bool, optional):
                    Whether to force the last checkpoint to be
                    saved regardless of interval. Default: True.
                sync_buffer (bool, optional): Whether to synchronize buffers in
                    different gpus. Default: False.
            """
            self.interval = interval
            self.by_epoch = by_epoch
            self.save_optimizer = save_optimizer
            self.max_keep_ckpts = max_keep_ckpts
            self.save_last = save_last
            self.args = kwargs
            self.sync_buffer = sync_buffer

        """Save checkpoints periodically."""

        def before_run(self, runner: BaseRunner):
            """This hook omits the setting process because it gets information
            from the ray session.

            Args:
                runner (:obj:`mmcv.runner.BaseRunner`):
                    The runner.
            """
            pass

        def before_train(self, runner) -> None:
            """This hook omits the setting process because it gets information
            from the ray session.

            Args:
                runner (:obj:`mmcv.runner.BaseRunner`):
                    The runner.
            """
            pass

        @master_only
        def _save_checkpoint(self, runner: BaseRunner) -> None:
            """Save checkpoints periodically.

            Args:
                runner (:obj:`mmcv.runner.BaseRunner`):
                    The runner to save checkpoints.
            """
            model = runner.model
            optimizer = runner.optimizer

            meta = dict(
                mmcv_version=mmcv.__version__,
                time=time.asctime(),
                epoch=runner.epoch + 1,
                iter=runner.iter + 1)
            if is_module_wrapper(model):
                model = model.module
            if hasattr(model, 'CLASSES') and model.CLASSES is not None:
                # save class name to the meta
                meta.update(CLASSES=model.CLASSES)
            checkpoint = {
                'meta': meta,
                'state_dict': weights_to_cpu(get_state_dict(model))
            }

            if isinstance(optimizer, Optimizer):
                checkpoint['optimizer'] = optimizer.state_dict()
            elif isinstance(optimizer, dict):
                checkpoint['optimizer'] = {}
                for name, optim in optimizer.items():
                    checkpoint['optimizer'][name] = optim.state_dict()

            step = (runner.epoch + 1) // self.interval
            if not self.by_epoch:
                step //= runner.iter + 1

            with tune.checkpoint_dir(step=step) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'ray_ckpt.pth')
                torch.save(checkpoint, path)
