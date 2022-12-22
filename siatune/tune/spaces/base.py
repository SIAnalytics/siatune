# Copyright (c) SI-Analytics. All rights reserved.
from abc import ABCMeta
from typing import Callable

import ray.tune as tune
from ray.tune.search.sample import Domain

from .builder import SPACES


class BaseSpace(metaclass=ABCMeta):
    """Base Space class."""
    sample: Callable = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @property
    def space(self) -> Domain:
        """Return the space."""
        return self.sample.__func__(**self.kwargs)


@SPACES.register_module()
class Uniform(BaseSpace):
    sample: Callable = tune.uniform


@SPACES.register_module()
class Quniform(BaseSpace):
    sample: Callable = tune.quniform


@SPACES.register_module()
class Loguniform(BaseSpace):
    sample: Callable = tune.loguniform


@SPACES.register_module()
class Qloguniform(BaseSpace):
    sample: Callable = tune.qloguniform


@SPACES.register_module()
class Randn(BaseSpace):
    sample: Callable = tune.randn


@SPACES.register_module()
class Qrandn(BaseSpace):
    sample: Callable = tune.qrandn


@SPACES.register_module()
class Randint(BaseSpace):
    sample: Callable = tune.randint


@SPACES.register_module()
class Qrandint(BaseSpace):
    sample: Callable = tune.qrandint


@SPACES.register_module()
class Lograndint(BaseSpace):
    sample: Callable = tune.lograndint


@SPACES.register_module()
class Qlograndint(BaseSpace):
    sample: Callable = tune.qlograndint
