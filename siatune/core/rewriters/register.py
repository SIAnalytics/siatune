# Copyright (c) SI-Analytics. All rights reserved.
from typing import Dict, List

from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class CustomHookRegister(BaseRewriter):
    """Register custom hooks."""

    def __init__(self,
                 key: str,
                 post_custom_hooks: List[Dict],
                 hk_key: str = 'custom_hooks') -> None:
        """Initialize the rewriter.

        Args:
            key (str): The key of the context.
            post_custom_hooks (List[Dict]): The custom hooks to be registered.
            hk_key (str): The key of the custom hooks.
        """
        # Re-register the hook in the new process.
        from siatune.core import hooks  # noqa F401

        self.post_custom_hooks = post_custom_hooks
        self.key = key
        self.hk_key = hk_key

    def __call__(self, context: Dict) -> Dict:
        """Register custom hooks.

        Args:
            context (Dict): The context to be rewritten.

        Returns:
            Dict: The context after rewriting.
        """

        custom_hooks = getattr(context[self.key], self.hk_key, []).copy()
        for custom_hook in self.post_custom_hooks:
            custom_hooks.append(custom_hook)
        setattr(context[self.key], self.hk_key, custom_hooks)
        return context
