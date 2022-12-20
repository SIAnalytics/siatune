# Copyright (c) SI-Analytics. All rights reserved.
import copy
import os.path as osp
from typing import Optional, Union

import mmcv
from ray.air.config import RunConfig
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner as RayTuner

from siatune.codebase import build_task_processor
from siatune.tune import (build_callback, build_scheduler, build_searcher,
                          build_space, build_stopper)


class Tuner:
    """Wrapper class of :class:`ray.tune.tuner.Tuner`.

    Args:
        task (dict): The trainable task to be tuned.
        work_dir (str): The working directory to save checkpoints. The logs
            will be saved in the subdirectory of `work_dir`.
        param_space (dict, optional): Search space of the tuning task.
        tune_cfg (dict, optional): Tuning algorithm specific configs
            except for `search_alg` and `scheduler`.
            Refer to :class:`ray.tune.tune_config.TuneConfig` for more info.
        searcher (dict, optional): Search algorithm for optimization.
            Default to random search.
            Refer to :module:`ray.tune.search` for more options.
        trial_scheduler (dict, optional): Scheduler for executing the trial.
            Default to FIFO scheduler.
            Refer to :module:`ray.tune.schedulers` for more options.
        stopper (dict, optional): Stop conditions to consider.
            Refer to :class:`ray.tune.stopper.Stopper` for more info.
        callbacks (dict | list, optional): Callbacks to invoke.
            Refer to :class:`ray.tune.callback.Callback` for more info.
        resume (str, optional): The experiment path to resume.
            Default to None.
        cfg (dict, optional) Full config. Default to None.
    """

    def __init__(
        self,
        task: dict,
        work_dir: str,
        param_space: Optional[dict] = None,
        tune_cfg: Optional[dict] = None,
        searcher: Optional[dict] = None,
        trial_scheduler: Optional[dict] = None,
        stopper: Optional[dict] = None,
        callbacks: Optional[Union[dict, list]] = None,
        resume: Optional[str] = None,
        cfg: Optional[dict] = None,
    ):
        task = build_task_processor(task)
        trainable = task.create_trainable()

        work_dir = osp.abspath(work_dir)
        mmcv.mkdir_or_exist(work_dir)

        if param_space is not None:
            param_space = build_space(param_space)

        tune_cfg = copy.deepcopy(tune_cfg or dict())

        if searcher is not None:
            searcher = build_searcher(searcher)

        if trial_scheduler is not None:
            trial_scheduler = build_scheduler(trial_scheduler)

        if stopper is not None:
            stopper = build_stopper(stopper)

        if callbacks is not None:
            if isinstance(callbacks, dict):
                callbacks = [callbacks]
            callbacks = [build_callback(callback) for callback in callbacks]

        self.resume = resume

        self.cfg = cfg

        self.tuner = RayTuner(
            trainable,
            param_space=dict(train_loop_config=param_space),
            tune_config=TuneConfig(
                search_alg=searcher, scheduler=trial_scheduler, **tune_cfg),
            run_config=RunConfig(
                local_dir=work_dir,
                stop=stopper,
                callbacks=callbacks,
                failure_config=None,  # todo
                sync_config=None,  # todo
                checkpoint_config=None,  # todo
            ),
        )

    @classmethod
    def from_cfg(cls, cfg: dict):
        cfg = copy.deepcopy(cfg)
        tuner = cls(
            task=cfg['task'],
            work_dir=cfg['work_dir'],
            param_space=cfg.get('space', None),
            tune_cfg=cfg.get('tune_cfg', None),
            searcher=cfg.get('searcher', None),
            trial_scheduler=cfg.get('trial_scheduler', None),
            stopper=cfg.get('stopper', None),
            callbacks=cfg.get('callbacks', None),
            resume=cfg.get('resume', None),
            cfg=cfg,
        )

        return tuner

    def tune(self):
        if self.resume is not None:
            self.tuner = RayTuner.restore(self.resume)

        return self.tuner.fit()
