import inspect
from typing import Callable

from mmcv.utils import Registry
from ray.tune import sample

from .base import BaseSpace

SPACES = Registry('spaces')


def register_space(space: Callable) -> None:

    @SPACES.register_module(name=space.__name__.capitalize())
    class _ImplicitSpace(BaseSpace):

        def __init__(self, *args, **kwargs):
            self._space = space(*args, **kwargs)


for space in dir(sample):
    if not inspect.isfunction(space):
        continue
    register_space(space)


def build_space(cfgs: dict) -> dict:
    return {k: SPACES.build(cfg) for k, cfg in cfgs}
