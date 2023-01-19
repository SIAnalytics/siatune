# Copyright (c) SI-Analytics. All rights reserved.
from mmengine.registry import Registry
from ray.tune.experiment import Trial

NAME_CREATORS = Registry('name creator')


@NAME_CREATORS.register_module()
def trial_id(trial: Trial) -> str:
    return trial.trial_id


@NAME_CREATORS.register_module()
def experiment_tag(trial: Trial) -> str:
    return trial.experiment_tag
