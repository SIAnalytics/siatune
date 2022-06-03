from typing import Dict

from mmcv.utils import Registry

from .base import BaseRewriter

REWRITERS = Registry('rewriters')


def build_rewriter(cfg: Dict) -> BaseRewriter:
    """Build a rewriter.

    Args:
        cfg (Dict): The config of the rewriter.

    Returns:
        BaseRewriter: The rewriter.
    """
    return REWRITERS.build(cfg)
