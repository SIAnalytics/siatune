# Copyright (c) SI-Analytics. All rights reserved.

import ray
from mmcv.utils import Config

from siatune.codebase import BaseTask
from siatune.tune import Tuner


def tune(task_processor: BaseTask, tune_config: Config,
         exp_name: str) -> ray.tune.ExperimentAnalysis:
    """Tune the task.

    Args:
        task_processor (BaseTask): The task processor.
            In each task processor, a targeted task is carried out.
        tune_config (Config): The config to control overall tuning.
        exp_name (str): The name of the experiment.

    Returns:
        ray.tune.ExperimentAnalysis: The analysis of the experiment.
    """

    trainable_cfg = tune_config.get('trainable', dict())
    trainable = task_processor.create_trainable(**trainable_cfg)

    tuner = Tuner.from_cfg(tune_config, trainable)
    return tuner.fit()
