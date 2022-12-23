# Copyright (c) SI-Analytics. All rights reserved.
from typing import Dict

from mmengine.registry import Registry

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
