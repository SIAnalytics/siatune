# Copyright (c) SI-Analytics. All rights reserved.
from typing import Callable, Union

import ray.tune as tune
from ray.tune.search.sample import Domain

from .base import BaseSpace
from .builder import SPACES


@SPACES.register_module()
class SampleFrom(BaseSpace):
    """Specify that tune should sample configuration values from this function.

    Args:
        func (str | Callable): An string or callable function
            to draw a sample from.
    """

    sample: Callable = tune.sample_from

    def __init__(self, func: Union[str, Callable]) -> None:
        if isinstance(func, str):
            assert func.startswith('lambda')
            func = eval(func)
        self.func = func

    @property
    def space(self) -> Domain:
        return self.sample.__func__(self.func)
