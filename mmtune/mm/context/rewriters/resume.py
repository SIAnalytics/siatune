from typing import Dict

from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class LoadFromCkpt(BaseRewriter):
    """Specifies the checkpoint for resuming training."""

    def __init__(
            self,
            key: str,
            load_key: str = 'load_from',
    ) -> None:
        """Initialize the rewriter.

        Args:
            key (str): The key where the instantiated cfg is stored.
            load_key(str): The key of load path.
        """
        self.key = key
        self.load_key = load_key

    def __call__(self, context: Dict) -> Dict:
        """Set with checkpoints specified by Ray.

        Args:
            context (Dict): The context to be rewritten.
        Returns:
            Dict: The context after rewriting.
        """
        setattr(context[self.key], self.load_key,
                context.pop('checkpoint_dir'))
        return context
