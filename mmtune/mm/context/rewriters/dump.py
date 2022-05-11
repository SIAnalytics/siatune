import tempfile
from os import path as osp

import ray

from .builder import REWRITERS


@REWRITERS.register_module()
class Dump:

    @staticmethod
    def get_temporary_path(file_name: str) -> str:
        temp_dir = tempfile.gettempdir()
        return osp.join(temp_dir, file_name)

    def __call__(self, context: dict) -> dict:
        cfg = context.pop('cfg')
        trial_id = ray.tune.get_trial_id()
        context['args'].config = self.get_temporary_path(f'{trial_id}.py')
        cfg.dump(context['args'].config)
        return context
