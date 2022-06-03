from os import path as osp

import ray

from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class AppendTrialIDtoPath(BaseRewriter):
    """Add the identifier of the tials to the workspace path to prevent the
    artifacts of each trial from being stored in the same path."""

    def __init__(self, key: str) -> None:
        """Initialize the rewriter.

        Args:
            key (str): The key to be changed.
        """
        self.key = key

    def __call__(self, context: dict) -> dict:
        """Give the workspace a different ID for each trial.

        Args:
            context (dict): The context to be rewritten.

        Returns:
            dict: The context after rewriting.
        """
        value = getattr(context['args'], self.key)
        setattr(context['args'], self.key,
                osp.join(value, ray.tune.get_trial_id()))
        return context
