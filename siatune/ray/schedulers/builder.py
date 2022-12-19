# Copyright (c) SI-Analytics. All rights reserved.
import inspect

from ray import tune
from ray.tune.schedulers import TrialScheduler

from siatune.mm.core import Config, Registry

TRIAL_SCHEDULERS = Registry('trial scheduler')
for v in set(tune.schedulers.SCHEDULER_IMPORT.values()):
    if not inspect.isclass(v):
        continue
    TRIAL_SCHEDULERS.register_module(module=v)


def build_scheduler(cfg: Config) -> TrialScheduler:
    """Build the scheduler from configs.

    Args:
        cfg (Config): The configs.
    Returns:
        tune.schedulers.TrialScheduler: The scheduler.
    """

    return TRIAL_SCHEDULERS.build(cfg)
