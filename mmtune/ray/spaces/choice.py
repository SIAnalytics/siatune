from numbers import Number
from typing import Callable, Optional, Sequence, Union

import ray.tune as tune

from mmtune.utils import ImmutableContainer
from .base import BaseSpace
from .builder import SPACES


@SPACES.register_module()
class Choice(BaseSpace):
    sample: Callable = tune.choice

    def __init__(self,
                 categories: Sequence,
                 alias: Optional[Sequence] = None,
                 use_container: bool = True):
        """Initialize Choice.

        Args:
            categories (Sequence): The categories.
            alias (Optional[Sequence]):
                A alias to be expressed. Defaults to None.
            use_container (bool):
                Whether to use containers. Defaults to True.
        """
        if alias is not None:
            assert len(categories) == len(alias)

        if use_container:
            aliases = alias or [None] * len(categories)
            categories = [
                ImmutableContainer(*it) for it in zip(categories, aliases)
            ]
        self.categories = categories

    @property
    def space(self) -> Union[Number, list]:
        return self.sample.__func__(self.categories)
