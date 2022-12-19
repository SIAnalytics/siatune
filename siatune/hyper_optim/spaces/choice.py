# Copyright (c) SI-Analytics. All rights reserved.
from typing import Callable, Optional, Sequence

import ray.tune as tune
from ray.tune.search.sample import Domain

from siatune.utils import ImmutableContainer
from .base import BaseSpace
from .builder import SPACES


@SPACES.register_module()
class Choice(BaseSpace):
    """Sample a categorical value.

    Args:
        categories (Sequence): The categories.
        alias (Sequence, optional): A alias to be expressed.
            Defaults to None.
    """

    sample: Callable = tune.choice

    def __init__(self,
                 categories: Sequence,
                 alias: Optional[Sequence] = None) -> None:
        if alias is not None:
            assert isinstance(alias, Sequence)
            assert len(categories) == len(alias)
        alias = alias or [None] * len(categories)
        categories = [ImmutableContainer(*it) for it in zip(categories, alias)]
        self.categories = categories

    @property
    def space(self) -> Domain:
        return self.sample.__func__(self.categories)
