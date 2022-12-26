# Copyright (c) SI-Analytics. All rights reserved.
import os
from os import path as osp

from ray.air import session

from siatune.utils import ref_raw_args
from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class RevertWorkSpace(BaseRewriter):

    def __init__(self, key='tune_launch_path'):
        self.key = key

    def __call__(self, context: dict) -> dict:
        os.chdir(context.pop(self.key))
        return context


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
                osp.join(value, session.get_trial_id()))
        return context


@REWRITERS.register_module()
class RawArgAppendTrialIDtoPath(BaseRewriter):

    def __init__(self, raw_arg_name: str = 'work-dir') -> None:
        """Initialize the rewriter.

        Args:
            arg_name (str): The arg_name to be changed.
        """
        self.raw_arg_name = raw_arg_name

    def __call__(self, context: dict) -> dict:
        """Give the workspace a different ID for each trial.

        Args:
            context (dict): The context to be rewritten.

        Returns:
            dict: The context after rewriting.
        """
        raw_args = context.get('raw_args')
        trial_id: str = session.get_trial_id()
        value, idx = ref_raw_args(raw_args, f'--{self.raw_arg_name}')
        assert len(idx) < 2
        if idx:
            raw_args[idx.pop()] = osp.join(value.pop(), trial_id)
        else:
            raw_args.extend([f'--{self.raw_arg_name}', trial_id])
        context.update(dict(raw_args=raw_args))
        return context
