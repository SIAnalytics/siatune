# Copyright (c) SI-Analytics. All rights reserved.
import argparse
from os import path as osp

from ray.air import session

from siatune.utils import reference_raw_args
from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class AttachTrialInfoToPath(BaseRewriter):
    """Add the identifier of the tials to the workspace path to prevent the
    artifacts of each trial from being stored in the same path."""
    arg_name: str = 'work_dir'
    raw_arg_name: str = '--work-dir'

    def __call__(self, context: dict) -> dict:
        """Give the workspace a different ID for each trial.

        Args:
            context (dict): The context to be rewritten.

        Returns:
            dict: The context after rewriting.
        """
        is_parsed = isinstance(context['args'], argparse.Namespace)
        if is_parsed:
            work_dir = getattr(context['args'], self.arg_name, '')
            if work_dir:
                work_dir = osp.join(work_dir, session.get_trial_id())
            else:
                work_dir = session.get_trial_dir()
            setattr(context['args'], self.arg_name, work_dir)
        else:
            work_dir, idx = reference_raw_args(context['args'],
                                               self.raw_arg_name)
            if idx:
                context['args'][idx.pop()] = osp.join(work_dir.pop(),
                                                      session.get_trial_id())
            else:
                context['args'].extend([
                    self.raw_arg_name,
                    osp.join('.', session.get_trial_dir())
                ])
        return context
