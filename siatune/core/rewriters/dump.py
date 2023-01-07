# Copyright (c) SI-Analytics. All rights reserved.
import argparse
import tempfile
from os import path as osp
from typing import Dict

from ray.air import session

from siatune.utils import dump_cfg
from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class Dump(BaseRewriter):

    arg_name: str = 'config'
    raw_arg_idx: int = 0
    """Dump the configs in the context as a file."""

    def __init__(self, key: str):
        """Inintialize the Dump class.

        Args:
            key (str): The key in the context.
        """
        self.key = key

    def _dump(self, cfg: Dict) -> str:
        tmpdir = tempfile.gettempdir()
        trial_id = session.get_trial_id()
        dmp_path = osp.join(tmpdir, f'{trial_id}.py')
        dump_cfg(cfg, dmp_path)
        return dmp_path

    def __call__(self, context: Dict) -> Dict:
        """Dump the configs in the context.

        Args:
            context (Dict): The context to be rewritten.

        Returns:
            Dict: The context after rewriting.
        """
        is_parsed = isinstance(context['args'], argparse.Namespace)
        dmp_path = self._dump(context.pop(self.key))
        if is_parsed:
            setattr(context['args'], self.arg_name, dmp_path)
        else:
            context['args'][self.raw_arg_idx] = dmp_path
        return context
