# Copyright (c) SI-Analytics. All rights reserved.
from typing import Mapping

from mmengine.registry import Registry

SPACES = Registry('space')


def build_space(cfg: dict) -> dict:
    """Build a space.

    Args:
        cfg (dict): The configurations of the space.

    Returns:
        dict: The instantiated space.
    """
    cfg = cfg.copy()
    for k, v in cfg.items():
        if isinstance(v, (int, str, bool, float)):
            continue
        elif isinstance(v, (list, tuple)):
            cfg[k] = type(v)(
                [build_space(_) if isinstance(_, dict) else _ for _ in v])
        elif isinstance(v, Mapping):
            cfg[k] = build_space(v)
            typ = cfg[k].get('type', '')
            if isinstance(typ, str) and typ in SPACES:
                cfg[k] = SPACES.build(cfg[k]).space
    return cfg
