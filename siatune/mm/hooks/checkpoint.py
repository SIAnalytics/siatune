# Copyright (c) SI-Analytics. All rights reserved.
import os
import time
from typing import Optional

import mmcv
import ray.tune as tune
import torch
from torch.optim import Optimizer

from siatune.mm.core import HOOKS, BaseRunner
from siatune.mm.core import CheckpointHook as _CheckpointHook
from siatune.mm.core import (get_state_dict, is_module_wrapper, master_only,
                             weights_to_cpu)


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
        """This hook omits the setting process because it gets information from
        the ray session.

        Args:
            runner (:obj:`mmcv.runner.BaseRunner`):
                The runner.
        """
        pass

    def before_train(self, runner) -> None:
        """This hook omits the setting process because it gets information from
        the ray session.

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
