# Copyright (c) SI-Analytics. All rights reserved.
import copy
import os.path as osp

from ray.air.config import RunConfig
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner as RayTuner

from siatune.ray import (build_callback, build_scheduler, build_searcher,
                         build_space, build_stopper)


class Tuner:
    """Wrapper class of :class:`ray.tune.tuner.Tuner`.

    Args:
        trainable (Callable):
        work_dir (str):
        param_space (dict, optional):
        tune_cfg (dict, optional):
            Refer to https://github.com/ray-project/ray/blob/ray-2.1.0/python/ray/tune/tune_config.py for details.  # noqa
        searcher (dict, optional):
        trial_scheduler (dict, optional):
        stopper (dict, optional):
        callbacks (list, optional):
    """

    def __init__(
        self,
        trainable,
        work_dir,
        param_space=None,
        tune_cfg=None,
        searcher=None,
        trial_scheduler=None,
        stopper=None,
        callbacks=None,
        resume=None,
    ):
        work_dir = osp.abspath(work_dir)

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
    def from_cfg(cls, cfg, trainable):
        cfg = copy.deepcopy(cfg)
        tuner = cls(
            trainable,
            work_dir=cfg['work_dir'],
            param_space=cfg.get('space', None),
            tune_cfg=cfg.get('tune_cfg', None),
            searcher=cfg.get('searcher', None),
            trial_scheduler=cfg.get('trial_scheduler', None),
            stopper=cfg.get('stopper', None),
            callbacks=cfg.get('callbacks', None),
            resume=cfg.get('resume', None),
        )

        return tuner

    def fit(self):
        if self.resume is not None:
            return self.tuner.restore(self.resume)

        return self.tuner.fit()
