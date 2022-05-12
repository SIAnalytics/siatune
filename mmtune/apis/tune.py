from os import path as osp

import mmcv
from mmcv.utils import Config
from ray import tune

from mmtune.mm import BaseTask
from mmtune.ray.algorithms import build_searcher
from mmtune.ray.schedulers import build_scheduler
from mmtune.ray.spaces import build_space
from mmtune.ray.stoppers import build_stopper

ARTIFACT_DIR_NAME = 'artifact'


def tune(task_processor: BaseTask, tune_config: Config,
         exp_name: str) -> tune.ExperimentAnalysis:
    trainable = task_processor.create_trainable(
        **getattr(tune_config, 'trainable', dict()))

    assert hasattr(tune_config, 'metric')
    assert hasattr(tune_config, 'mode') and tune_config.mode in ['min', 'max']

    tune_artifact_dir = osp.join(task_processor.work_dir, ARTIFACT_DIR_NAME)
    mmcv.mkdir_or_exist(tune_artifact_dir)

    return tune.run(
        trainable,
        metric=tune_config.metric,
        mode=tune_config.mode,
        name=exp_name,
        resources_per_trial=dict(
            cpu=task_processor.ARGS.num_workers *  # noqa W504
            task_processor.ARGS.num_cpus_per_worker,
            gpu=task_processor.ARGS.num_workers *  # noqa W504
            task_processor.ARGS.num_gpus_per_worker),
        stop=build_stopper(getattr(tune_config, 'stop', None)),
        config=build_space(getattr(tune_config, 'space', None)),
        num_samples=getattr(tune_config, 'num_samples', -1),
        local_dir=tune_artifact_dir,
        searcher=build_searcher(getattr(tune_config, 'searcher', None)),
        scheduler=build_scheduler(getattr(tune_config, 'scheduler', None)),
    )
