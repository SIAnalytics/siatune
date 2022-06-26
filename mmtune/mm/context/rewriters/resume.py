from os import path as osp
from typing import Dict

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
