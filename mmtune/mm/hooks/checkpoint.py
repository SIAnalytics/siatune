import os
import time

import mmcv
import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS
from mmcv.runner.checkpoint import get_state_dict, weights_to_cpu
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import CheckpointHook as _CheckpointHook
from ray.tune.integration.torch import distributed_checkpoint_dir


@HOOKS.register_module()
class RayCheckpointHook(_CheckpointHook):
    def get_iter(self, runner, inner_iter=False):
        """Get the current training iteration step."""
        if self.by_epoch and inner_iter:
            current_iter = runner.inner_iter + 1
        else:
            current_iter = runner.iter + 1
        return current_iter

    @master_only
    def _save_checkpoint(self, runner):
        model = runner.model
        checkpoint_dir = self.out_dir

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
