from numbers import Number
from typing import Callable, Union

import ray.tune as tune

from .base import BaseSpace
from .builder import SPACES


@SPACES.register_module()
class SampleFrom(BaseSpace):
    sample: Callable = tune.sample_from

    def __init__(self, func: Union[str, Callable], imports=None):
        if isinstance(func, str):
            func = eval(func)
        self.func = func
        self.imports = imports or []

    @property
    def space(self) -> Union[Number, list]:
        for module in self.imports:
            exec(f'import {module}')
        return self.sample.__func__(self.func)
