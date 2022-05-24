from typing import Dict

from mmcv.utils import Registry

REWRITERS = Registry('rewriters')


def build_rewriter(cfg: Dict) -> object:
    return REWRITERS.build(cfg)
