import inspect
from typing import Callable, Dict

from mmcv.utils import Registry
from ray.tune import sample

from .base import BaseSpace

SPACES = Registry('spaces')


def _register_space(space: Callable) -> None:
    """Register a space.

    Args:
        space (Callable): The space to register.
    """

    @SPACES.register_module(name=space.__name__.capitalize())
    class _ImplicitSpace(BaseSpace):

        def __init__(self, *args, **kwargs):
            self._space = space(*args, **kwargs)


for space_name in dir(sample):
    space = getattr(sample, space_name)
    if not inspect.isfunction(space):
        continue
    _register_space(space)


def build_space(cfgs: Dict) -> Dict:
    """Build a space.

    Args:
        cfgs (Dict): The configurations of the space.

    Returns:
        Dict: The instantiated space.
    """

    return {key: SPACES.build(cfg).space for key, cfg in cfgs.items()}
