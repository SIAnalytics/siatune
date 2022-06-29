from typing import List, Optional

from ray.tune import grid_search

from mmtune.utils import ImmutableContainer
from .base import BaseSpace
from .builder import SPACES


@SPACES.register_module()
class GridSearch(BaseSpace):
    """Grid search over a value."""

    def __init__(self,
                 values: List,
                 alias: Optional[List] = None,
                 use_container: bool = True):
        """Initialize Grid searcher.

        Args:
            values (Sequence): An iterable whose parameters will be gridded.
            alias (Optional[Sequence]):
                A alias to be expressed. Defaults to None.
            use_container (bool):
                Whether to use containers. Defaults to True.
        """
        if alias is not None:
            assert len(values) == len(alias)
        values = [
            ImmutableContainer(v, None if alias is None else alias[idx])
            if use_container else v for idx, v in enumerate(values)
        ]
        setattr(self, '_space', grid_search(values))
