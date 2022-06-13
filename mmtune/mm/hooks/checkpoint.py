import os
import time

import mmcv
import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, BaseRunner
from mmcv.runner.checkpoint import get_state_dict, weights_to_cpu
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import CheckpointHook as _CheckpointHook
from ray.tune.integration.torch import distributed_checkpoint_dir
from typing import Optional

@HOOKS.register_module()
class RayCheckpointHook(_CheckpointHook):
    def __init__(self,
                 interval: int=-1,
                 by_epoch:bool=True,
                 save_optimizer:bool=True,
                 max_keep_ckpts:int=-1,
                 save_last:bool=True,
                 sync_buffer:Optional[bool]=False,
                 file_client_args:Optional[dict]=None,
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
            save_last (bool, optional): Whether to force the last checkpoint to be
                saved regardless of interval. Default: True.
            sync_buffer (bool, optional): Whether to synchronize buffers in
                different gpus. Default: False.
            file_client_args (dict, optional): Arguments to instantiate a
                FileClient. See :class:`mmcv.fileio.FileClient` for details.
                Default: None.
                `New in version 1.3.16.` 
        """  
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs
        self.sync_buffer = sync_buffer
        self.file_client_args = file_client_args 

    """Save checkpoints periodically."""

    def get_iter(self, runner: BaseRunner, inner_iter: bool = False):
        """Get the current iteration.

        Args:
            runner (:obj:`mmcv.runner.BaseRunner`):
                The runner to get the current iteration.
            inner_iter (bool):
                Whether to get the inner iteration.
        """

        if self.by_epoch and inner_iter:
            current_iter = runner.inner_iter + 1
        else:
            current_iter = runner.iter + 1
        return current_iter

    @master_only
    def _save_checkpoint(self, runner: BaseRunner) -> None:
        """Save checkpoints periodically.

        Args:
            runner (:obj:`mmcv.runner.BaseRunner`):
                The runner to save checkpoints.
        """
        model = runner.model

        meta = dict(mmcv_version=mmcv.__version__, time=time.asctime())
        if is_module_wrapper(model):
            model = model.module
        if hasattr(model, 'CLASSES') and model.CLASSES is not None:
            # save class name to the meta
            meta.update(CLASSES=model.CLASSES)
        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(get_state_dict(model))
        }

        with distributed_checkpoint_dir(
                step=self.get_iter(runner)) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'ray_checkpoint.pth')
            torch.save(checkpoint, path)
