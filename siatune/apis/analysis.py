# Copyright (c) SI-Analytics. All rights reserved.
import time
from os import path as osp
from pprint import pformat
from typing import Optional

from mmengine.config import Config
from ray.tune import ResultGrid

from siatune.utils import ImmutableContainer, dump_cfg
from siatune.utils.logger import get_root_logger


def log_analysis(results: ResultGrid,
                 tune_config: Config,
                 task_config: Optional[Config] = None,
                 log_dir: Optional[str] = None) -> None:
    """Log the analysis of the experiment.

    Args:
        results (ResultGrid): Experiment results of `Tuner.fit()`.
        tune_config (Config): The tune config.
        task_config (Optional[Config]): The task config. Defaults to None.
        log_dir (Optional[str]): The log dir. Defaults to None.
    """
    log_dir = log_dir or tune_config.work_dir

    dump_cfg(tune_config, osp.join(log_dir, 'tune_config.py'))

    if task_config is not None:
        dump_cfg(task_config, osp.join(log_dir, 'task_config.py'))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_root_logger(
        'siatune', log_file=osp.join(log_dir, f'{timestamp}.log'))

    result = results.get_best_result()
    logger.info(f'Best Result: \n'
                f'{pformat(ImmutableContainer.decouple(result))}')
    logger.info(f'Best Hyperparam: \n'
                f'{pformat(ImmutableContainer.decouple(result.config))}')
    logger.info(f'Best Logdir: {result.log_dir}')
