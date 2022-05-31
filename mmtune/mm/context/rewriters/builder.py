from typing import Dict

from mmcv.utils import Registry

REWRITERS = Registry('rewriters')


def build_rewriter(cfg: Dict) -> object:
    """Build a rewriter.

    Args:
        cfg (Dict): The config of the rewriter.

    Returns:
        object: The rewriter.
    """
    return REWRITERS.build(cfg)
