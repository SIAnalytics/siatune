from os import path as osp

import ray

from .builder import REWRITERS


@REWRITERS.register_module()
class PathJoinTrialId:

    def __init__(self, key: str) -> None:
        """Initialize the rewriter.

        Args:
            key (str): The key to be changed.
        """
        self.key = key

    def __call__(self, context: dict) -> dict:
        """Give the workspace a different ID for each trial.

        Args:
            context (dict): The context of the trial.

        Returns:
            dict: The context of the trial.
        """
        value = getattr(context['args'], self.key)
        setattr(context['args'], self.key,
                osp.join(value, ray.tune.get_trial_id()))
        return context
