# Copyright (c) SI-Analytics. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict

import torch
from ray.air import checkpoint, session

from siatune.utils import ref_raw_args
from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class ResumeFromCkpt(BaseRewriter):
    """Specifies the checkpoint for resuming training."""
    arg_name: str = 'resume_from'
    raw_arg_name: str = '--resume-from'

    def _dump(self, ckpt: checkpoint.Checkpoint) -> str:
        tmp_dir = tempfile.gettempdir()
        ckpt_path = osp.join(tmp_dir, f'{session.get_trial_id()}.pth')
        torch.save(ckpt.to_dict(), ckpt_path)
        return ckpt_path

    def __call__(self, context: Dict) -> Dict:
        """Set with checkpoints specified by Ray.

        Args:
            context (Dict): The context to be rewritten.
        Returns:
            Dict: The context after rewriting.
        """
        ckpt: checkpoint.Checkpoint = session.get_checkpoint()
        if not ckpt:
            return context
        ckpt_path = self._dump(ckpt)
        is_parsed = not isinstance(context['args'], list)
        if is_parsed:
            setattr(context['args'], self.arg_name, ckpt_path)
        else:
            _, idx = ref_raw_args(context['args'], self.raw_arg_name)
            assert len(idx) < 2
            if idx:
                context['args'][idx.pop()] = ckpt_path
            else:
                context['args'].extend([self.raw_arg_name, ckpt_path])
        return context
