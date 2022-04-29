from os import path as osp

import ray


class SetEnv:

    def __call__(self, context: dict) -> dict:
        context['args'].work_dir = osp.join(context['args'].work_dir,
                                            ray.tune.get_trial_id())
        return context
