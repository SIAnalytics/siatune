from typing import Optional, Sequence

from ray.tune.sample import choice

from mmtune.utils import ImmutableContainer
from .base import BaseSpace
from .builder import SPACE


@SPACE.register_module()
class Choice(BaseSpace):

    def __init__(self,
                 categories: Sequence,
                 alias: Optional[Sequence] = None,
                 use_container: bool = True):
        if alias is not None:
            assert len(categories) == len(alias)
        categories = [
            ImmutableContainer(c, None if alias is None else alias[idx])
            if use_container else c for idx, c in enumerate(categories)
        ]
        self._space = choice(categories)
