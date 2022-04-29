import inspect

from mmcv.utils import Config, Registry
from ray import tune

STOPPER = Registry('stopper')
for stopper in dir(tune.stopper):
    if not stopper.endswith('Stopper'):
        continue
    stopper_cls = getattr(tune.stopper, stopper)
    if not inspect.isclass(stopper_cls):
        continue
    STOPPER.register_module(stopper_cls)


def build_stopper(cfg: Config) -> tune.stopper:
    return STOPPER.build(cfg)
