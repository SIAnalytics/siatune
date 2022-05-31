from mmtune.utils import ImmutableContainer
from .builder import REWRITERS


@REWRITERS.register_module()
class Decouple:
    """Decouple the configs in the immutable container."""

    def __init__(self, key: str) -> None:
        """
        Args:
            key (str): The key of the configs in the immutable container.
        """
        self.key = key

    def __call__(self, context: dict) -> dict:
        """Decouple the configs in the immutable container.

        Args:
            context (dict): The context to be rewritten.

        Returns:
            dict: The context after rewriting.
        """
        context[self.key] = ImmutableContainer.decouple(context[self.key])
        return context
