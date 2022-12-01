# Copyright (c) SI-Analytics. All rights reserved.
from os import path as osp
from typing import Dict

from .base import BaseRewriter
from .builder import REWRITERS

from mmcv.utils import Config
    
@REWRITERS.register_module()
class RestoreCkptToLoad(BaseRewriter):
    """Specifies the checkpoint for restoring training.
    Rewriter classes required for PBT-based algorithms. 
    Independent of optimizer state.
    """

    ckpt_base_name = 'ray_ckpt.pth'

    def __init__(self, key: str) -> None:
        """Initialize the rewriter.

        Args:
            key (str): The key where the instantiated cfg is stored.
            arg_name (str): The key in the argparse namespace.
        """
        self.key = key

    def __call__(self, context: Dict) -> Dict:
        """Set with checkpoints specified by Ray.

        Args:
            context (Dict): The context to be rewritten.
        Returns:
            Dict: The context after rewriting.
        """
        if context.get('checkpoint_dir') is not None:
            assert isinstance(context[self.key], Config)
            context[self.key].load_from = osp.join(context.pop('checkpoint_dir'), self.ckpt_base_name)
        return context


@REWRITERS.register_module()
class RestoreCkptToResume(BaseRewriter):
    """Specifies the checkpoint for restoring training.
    Rewriter classes required for PBT-based algorithms. 
    Dependent of optimizer state.
    """

    ckpt_base_name = 'ray_ckpt.pth'

    def __init__(self, arg_name: str = 'resume_from') -> None:
        """Initialize the rewriter.

        Args:
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
                osp.join(context.pop('checkpoint_dir'), self.ckpt_base_name))
        return context
