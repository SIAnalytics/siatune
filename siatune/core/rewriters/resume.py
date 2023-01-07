# Copyright (c) SI-Analytics. All rights reserved.
import argparse
from typing import Dict

from ray.air import session

from siatune.utils import reference_raw_args
from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class ResumeFromCkpt(BaseRewriter):
    """Specifies the checkpoint for resuming training."""
    arg_name: str = 'resume_from'
    raw_arg_name: str = '--resume-from'

    def __call__(self, context: Dict) -> Dict:
        """Set with checkpoints specified by Ray.

        Args:
            context (Dict): The context to be rewritten.
        Returns:
            Dict: The context after rewriting.
        """
        ckpt = session.get_checkpoint()
        if not ckpt:
            return context
        ckpt_path = ckpt.to_dict().get('path')
        is_parsed = isinstance(context['args'], argparse.Namespace)
        if is_parsed:
            setattr(context['args'], self.arg_name, ckpt_path)
        else:
            _, idx = reference_raw_args(context['args'], self.raw_arg_name)
            assert len(idx) < 2
            if idx:
                context['args'][idx.pop()] = ckpt_path
            else:
                context['args'].extend([self.raw_arg_name, ckpt_path])
        return context
