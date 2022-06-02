from typing import Any, Optional

from mmtune.utils import ImmutableContainer
from .base import BaseSpace
from .builder import SPACES


@SPACES.register_module()
class Constant(BaseSpace):
    """Constant space."""

    def __init__(self,
                 value: Any,
                 alias: Optional[str] = None,
                 use_container: bool = True) -> None:
        """Initialize constant space.

        Args:
            value (Any): The value.
            alias (Optional[str], optional):
                A alias to be expressed. Defaults to None.
            use_container (bool, optional):
                Whether to use containers. Defaults to True.
        """
        if use_container:
            value = ImmutableContainer(value, alias)
        self._space = value
