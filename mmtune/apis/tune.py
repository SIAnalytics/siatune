from os import path as osp

import mmcv
import ray
from mmcv.utils import Config

from mmtune.mm.tasks import BaseTask
from mmtune.ray.schedulers import build_scheduler
from mmtune.ray.searchers import build_searcher
from mmtune.ray.spaces import build_space
from mmtune.ray.stoppers import build_stopper

ARTIFACT_DIR_NAME = 'artifact'


def tune(task_processor: BaseTask, tune_config: Config,
         exp_name: str) -> ray.tune.ExperimentAnalysis:
    trainable = task_processor.create_trainable(
        **getattr(tune_config, 'trainable', dict()))

    assert hasattr(tune_config, 'metric')
    assert hasattr(tune_config, 'mode') and tune_config.mode in ['min', 'mode']

    tune_artifact_dir = osp.join(task_processor.ARGS.work_dir,
                                 ARTIFACT_DIR_NAME)
    mmcv.mkdir_or_exist(tune_artifact_dir)

    return ray.tune.run(
        trainable,
        metric=tune_config.metric,
        mode=tune_config.mode,
        name=exp_name,
        resources_per_trial=dict(
            cpu=task_processor.ARGS.num_workers *  # noqa W504
            task_processor.ARGS.num_cpus_per_worker,
            gpu=task_processor.ARGS.num_workers *  # noqa W504
            task_processor.ARGS.num_gpus_per_worker),
        stop=build_stopper(tune_config.stop)
        if hasattr(tune_config, 'stop') else None,
        config=build_space(tune_config.space)
        if hasattr(tune_config, 'space') else None,
        num_samples=getattr(tune_config, 'num_samples', -1),
        local_dir=tune_artifact_dir,
        search_alg=build_searcher(tune_config.searcher) if hasattr(
            tune_config, 'searcher') else None,
        scheduler=build_scheduler(tune_config.scheduler) if hasattr(
            tune_config, 'scheduler') else None,
    )
