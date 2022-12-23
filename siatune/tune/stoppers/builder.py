# Copyright (c) SI-Analytics. All rights reserved.
import inspect

from mmengine.config import Config
from mmengine.registry import Registry
from ray import tune

STOPPERS = Registry('stopper')
for stopper in dir(tune.stopper):
    if not stopper.endswith('Stopper'):
        continue
    stopper_cls = getattr(tune.stopper, stopper)
    if not inspect.isclass(stopper_cls):
        continue
    STOPPERS._register_module(stopper_cls)


def build_stopper(cfg: Config) -> tune.stopper:
    """Build a stopper.

    Args:
        cfg (Config): The configuration of the stopper.

    Returns:
        tune.stopper: The instantiated stopper.
    """
    return STOPPERS.build(cfg)
