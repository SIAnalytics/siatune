import tempfile
from os import path as osp
from typing import Dict

import ray

from mmtune.utils import dump_cfg
from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class Dump(BaseRewriter):
    """Dump the configs in the context."""

    def __init__(self, ctx_key: str, arg_key: str):
        """Inintialize the Dump class.

        Args:
            ctx_key (str): The key in the context.
            arg_key (str): The key in the argparse namespace.
        """
        self.ctx_key = ctx_key
        self.arg_key = arg_key

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
        cfg = context.pop(self.ctx_key)
        trial_id = ray.tune.get_trial_id()
        tmp_path = self.get_temporary_path(f'{trial_id}.py')
        setattr(context.get('args'), self.arg_key, tmp_path)
        dump_cfg(cfg, tmp_path)
        return context
