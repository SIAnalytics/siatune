from os import path as osp

import mmcv
import ray
from mmcv.utils import Config

from mmtune.mm.tasks import BaseTask
from mmtune.ray.schedulers import build_scheduler
from mmtune.ray.searchers import build_searcher
from mmtune.ray.spaces import build_space
from mmtune.ray.stoppers import build_stopper


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

    assert hasattr(tune_config, 'metric')
    assert hasattr(tune_config, 'mode') and tune_config.mode in ['min', 'max']

    tune_artifact_dir = osp.join(tune_config.work_dir, 'artifact')
    mmcv.mkdir_or_exist(tune_artifact_dir)

    stopper = tune_config.get('stop', None)
    if stopper is not None:
        stopper = build_stopper(stopper)

    space = tune_config.get('space', None)
    if space is not None:
        space = build_space(space)

    resources_per_trial = None
    if not hasattr(trainable, 'default_resource_request'):
        num_workers = trainable_cfg.get('num_workers', 1)
        num_gpus_per_worker = trainable_cfg.get('num_gpus_per_worker', 1)
        num_cpus_per_worker = trainable_cfg.get('num_cpus_per_worker', 1)
        resources_per_trial = dict(
            gpu=num_workers * num_gpus_per_worker,
            cpu=num_workers * num_cpus_per_worker)

    searcher = tune_config.get('searcher', None)
    if searcher is not None:
        searcher = build_searcher(searcher)

    scheduler = tune_config.get('scheduler', None)
    if scheduler is not None:
        scheduler = build_scheduler(scheduler)

    return ray.tune.run(
        trainable,
        name=exp_name,
        metric=tune_config.metric,
        mode=tune_config.mode,
        stop=stopper,
        config=space,
        resources_per_trial=resources_per_trial,
        num_samples=tune_config.get('num_samples', -1),
        local_dir=tune_artifact_dir,
        search_alg=searcher,
        scheduler=scheduler,
        raise_on_failed_trial=tune_config.get('raise_on_failed_trial', False))
