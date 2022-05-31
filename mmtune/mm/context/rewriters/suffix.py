from os import path as osp

import ray

from .builder import REWRITERS


@REWRITERS.register_module()
class SuffixTrialId:

    def __init__(self, key):
        self.key = key

    def __call__(self, context: dict) -> dict:
        context[self.key] = osp.join(context[self.key], ray.get_trial_id())
        return context
