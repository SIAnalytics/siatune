# Copyright (c) SI-Analytics. All rights reserved.
import inspect

from mmengine.config import Config
from mmengine.registry import Registry
from ray import tune
from ray.tune.schedulers import TrialScheduler

TRIAL_SCHEDULERS = Registry('trial scheduler')

# Dynamically import scheduler
# Refer to https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers/__init__.py  # noqa
for cls in set(tune.schedulers.SCHEDULER_IMPORT.values()):
    if not inspect.isclass(cls):
        continue
    TRIAL_SCHEDULERS.register_module(module=cls)


def build_scheduler(cfg: Config) -> TrialScheduler:
    """Build the scheduler from configs.

    Args:
        cfg (Config): The configs.
    Returns:
        tune.schedulers.TrialScheduler: The scheduler.
    """

    return TRIAL_SCHEDULERS.build(cfg)
