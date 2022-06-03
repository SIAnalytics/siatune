from typing import Dict

from mmtune.utils import ImmutableContainer
from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class Decouple(BaseRewriter):
    """Decouple the configs in the immutable container."""

    def __init__(self, key: str) -> None:
        """
        Args:
            key (str): The key of the configs in the immutable container.
        """
        self.key = key

    def __call__(self, context: Dict) -> Dict:
        """Decouple the configs in the immutable container.

        Args:
            context (Dict): The context to be rewritten.

        Returns:
            Dict: The context after rewriting.
        """
        context[self.key] = ImmutableContainer.decouple(context[self.key])
        return context
