# Copyright (c) SI-Analytics. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict

import ray

from siatune.utils import dump_cfg
from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class Dump(BaseRewriter):
    """Dump the configs in the context as a file."""

    def __init__(self, key: str, arg_name: str):
        """Inintialize the Dump class.

        Args:
            key (str): The key in the context.
            arg_name (str): The key in the argparse namespace.
        """
        self.key = key
        self.arg_name = arg_name

    @staticmethod
    def get_temporary_path(file_name: str) -> str:
        """Get the temporary path.

        Args:
            file_name (str): The name of the file.

        Returns:
            str: The temporary path.
        """
        return osp.join(tempfile.gettempdir(), file_name)

    def __call__(self, context: Dict) -> Dict:
        """Dump the configs in the context.

        Args:
            context (Dict): The context to be rewritten.

        Returns:
            Dict: The context after rewriting.
        """
        cfg = context.pop(self.key)
        trial_id = ray.tune.get_trial_id()
        tmp_path = self.get_temporary_path(f'{trial_id}.py')
        setattr(context.get('args'), self.arg_name, tmp_path)
        dump_cfg(cfg, tmp_path)
        return context
