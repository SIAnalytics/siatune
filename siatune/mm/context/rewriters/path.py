# Copyright (c) SI-Analytics. All rights reserved.
from os import path as osp

import ray

from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class AppendTrialIDtoPath(BaseRewriter):
    """Add the identifier of the tials to the workspace path to prevent the
    artifacts of each trial from being stored in the same path."""

    def __init__(self, arg_name: str) -> None:
        """Initialize the rewriter.

        Args:
            arg_name (str): The arg_name to be changed.
        """
        self.arg_name = arg_name

    def __call__(self, context: dict) -> dict:
        """Give the workspace a different ID for each trial.

        Args:
            context (dict): The context to be rewritten.

        Returns:
            dict: The context after rewriting.
        """
        value = getattr(context['args'], self.arg_name)
        setattr(context['args'], self.arg_name,
                osp.join(value, ray.tune.get_trial_id()))
        return context


@REWRITERS.register_module()
class RawArgAppendTrialIDtoPath(BaseRewriter):

    def __init__(self, arg_name: str = '--work-dir') -> None:
        """Initialize the rewriter.

        Args:
            arg_name (str): The arg_name to be changed.
        """
        self.arg_name = arg_name

    def __call__(self, context: dict) -> dict:
        """Give the workspace a different ID for each trial.

        Args:
            context (dict): The context to be rewritten.

        Returns:
            dict: The context after rewriting.
        """
        raw_args = context['raw_args']
        trial_id: str = ray.tune.get_trial_id()
        if self.arg_name in raw_args:
            idx = raw_args.index(self.arg_name) + 1
            assert idx < len(raw_args)
            parent = raw_args[idx]
            assert not parent.startswith('--')
            raw_args[idx] = osp.join(parent, trial_id)
        else:
            raw_args.extend([self.arg_name, trial_id])
        return context
