# Copyright (c) SI-Analytics. All rights reserved.
from typing import Callable, Optional, Sequence

import ray.tune as tune

from siatune.utils import ImmutableContainer
from .base import BaseSpace
from .builder import SPACES


@SPACES.register_module()
class GridSearch(BaseSpace):
    """Grid search over a value.

    Args:
        values (Sequence): An iterable whose parameters will be gridded.
        alias (Sequence, optional): A alias to be expressed.
            Defaults to None.
    """

    sample: Callable = tune.grid_search

    def __init__(self,
                 values: Sequence,
                 alias: Optional[Sequence] = None) -> None:
        if alias is not None:
            assert len(values) == len(alias)
            values = [ImmutableContainer(*it) for it in zip(values, alias)]
        self.values = values

    @property
    def space(self) -> dict:
        return self.sample.__func__(self.values)
