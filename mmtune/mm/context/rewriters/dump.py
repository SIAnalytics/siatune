import tempfile
from os import path as osp

import ray

from .builder import REWRITERS


@REWRITERS.register_module()
class Dump:
    """Dump the configs in the context."""

    def __init__(self, ctx_key: str, arg_key: str):
        """inintialize the Dump class.

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
        temp_dir = tempfile.gettempdir()
        return osp.join(temp_dir, file_name)

    def __call__(self, context: dict) -> dict:
        """Dump the configs in the context.

        Args:
            context (dict): The context to be rewritten.

        Returns:
            dict: The context after rewriting.
        """
        cfg = context.pop(self.ctx_key)
        trial_id = ray.tune.get_trial_id()
        tmp_path = self.get_temporary_path(f'{trial_id}.py')
        setattr(context.get('args'), self.arg_key, tmp_path)
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(cfg.pretty_text)
        return context
