import tempfile
from os import path as osp

import ray

from .builder import REWRITERS


@REWRITERS.register_module()
class Dump:

    def __init__(self, ctx_key: str, arg_key: str):
        self.ctx_key = ctx_key
        self.arg_key = arg_key

    @staticmethod
    def get_temporary_path(file_name: str) -> str:
        temp_dir = tempfile.gettempdir()
        return osp.join(temp_dir, file_name)

    def __call__(self, context: dict) -> dict:
        cfg = context.pop('cfg')
        trial_id = ray.tune.get_trial_id()
        tmp_path = self.get_temporary_path(f'{trial_id}.py')
        setattr(context.get(self.ctx_key), self.arg_key, tmp_path)
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(cfg.pretty_text)
        return context
