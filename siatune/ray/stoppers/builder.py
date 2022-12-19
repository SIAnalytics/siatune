# Copyright (c) SI-Analytics. All rights reserved.
import inspect

from ray import tune

from siatune.mm.core import MMENGINE_BASED, Config, Registry

STOPPERS = Registry('stoppers')
for stopper in dir(tune.stopper):
    if not stopper.endswith('Stopper'):
        continue
    stopper_cls = getattr(tune.stopper, stopper)
    if not inspect.isclass(stopper_cls):
        continue
    if MMENGINE_BASED:
        STOPPERS._register_module(stopper_cls)
    else:
        STOPPERS.register_module(stopper_cls)


def build_stopper(cfg: Config) -> tune.stopper:
    """Build a stopper.

    Args:
        cfg (Config): The configuration of the stopper.

    Returns:
        tune.stopper: The instantiated stopper.
    """
    return STOPPERS.build(cfg)
