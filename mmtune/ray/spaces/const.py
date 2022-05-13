from typing import Any, Optional

from mmtune.utils import ImmutableContainer
from .base import BaseSpace
from .builder import SPACES


@SPACES.register_module()
class Constant(BaseSpace):

    def __init__(self,
                 value: Any,
                 alias: Optional[str] = None,
                 use_container: bool = True):
        if use_container:
            value = ImmutableContainer(value, alias)
        self._space = value
