from os import path as osp

import ray

from .builder import REWRITERS


@REWRITERS.register_module()
class SuffixTrialId:

    def __init__(self, key):
        self.key = key

    def __call__(self, context: dict) -> dict:
        value = getattr(context['args'], self.key)
        setattr(context['args'], self.key,
                osp.join(value, ray.tune.get_trial_id()))
        return context
