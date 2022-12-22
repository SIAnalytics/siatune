# Copyright (c) SI-Analytics. All rights reserved.
from os import path as osp
from typing import Dict, List

from ray.air import session

from siatune.utils import ref_raw_args
from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class ResumeFromCkpt(BaseRewriter):
    """Specifies the checkpoint for resuming training."""

    def __init__(self, arg_name: str = 'resume_from') -> None:
        """Initialize the rewriter.

        Args:
            key (str): The key where the instantiated cfg is stored.
            arg_name (str): The key in the argparse namespace.
        """
        self.arg_name = arg_name

    def __call__(self, context: Dict) -> Dict:
        """Set with checkpoints specified by Ray.

        Args:
            context (Dict): The context to be rewritten.
        Returns:
            Dict: The context after rewriting.
        """
        if context.get('checkpoint_dir') is not None:
            setattr(
                context.get('args'), self.arg_name,
                osp.join(context.pop('checkpoint_dir'), 'ray_ckpt.pth'))
        return context


@REWRITERS.register_module()
class RawArgResumeFromCkpt(BaseRewriter):
    """Specifies the checkpoint for resuming training."""
    ckpt_name = 'ray_ckpt.pth'

    def __init__(self, arg_name: str = 'resume_from') -> None:
        """Initialize the rewriter.

        Args:
            key (str): The key where the instantiated cfg is stored.
            arg_name (str): The key in the argparse namespace.
        """
        self.arg_name = arg_name

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
        raw_args: List[str] = context.get('raw_args', [])
        with ckpt.as_directory() as loaded_checkpoint_dir:
            _, idx = ref_raw_args(raw_args, f'--{self.arg_name}')
            assert len(idx) < 2

            if idx:
                raw_args[idx.pop()] = osp.join(loaded_checkpoint_dir,
                                               self.ckpt_name)
            else:
                raw_args.extend([
                    f'--{self.arg_name}',
                    osp.join(loaded_checkpoint_dir, self.ckpt_name)
                ])
        context.update(dict(raw_args=raw_args))
        return context
