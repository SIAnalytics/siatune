import inspect

from mmcv.utils import Config, Registry
from ray import tune

SCHEDULER = Registry('scheduler')
for v in set(tune.schedulers.SCHEDULER_IMPORT.values()):
    if not inspect.isclass(v):
        continue
    SCHEDULER.register_module(module=v)


def build_scheduler(cfg: Config) -> tune.schedulers.TrialScheduler:
    return SCHEDULER.build(cfg)
