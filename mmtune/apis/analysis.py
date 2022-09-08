# Copyright (c) SI-Analytics. All rights reserved.
import time
from os import path as osp
from pprint import pformat
from typing import Optional

from mmcv.utils import Config, get_logger
from ray import tune

from mmtune.utils import ImmutableContainer, dump_cfg


def log_analysis(analysis: tune.ExperimentAnalysis,
                 tune_config: Config,
                 task_config: Optional[Config] = None,
                 log_dir: Optional[str] = None) -> None:
    """Log the analysis of the experiment.

    Args:
        analysis (tune.ExperimentAnalysis): The analysis of the experiment.
        tune_config (Config): The tune config.
        task_config (Optional[Config]): The task config. Defaults to None.
        log_dir (Optional[str]): The log dir. Defaults to None.
    """
    log_dir = log_dir or tune_config.work_dir

    dump_cfg(tune_config, osp.join(log_dir, 'tune_config.py'))

    if task_config is not None:
        dump_cfg(task_config, osp.join(log_dir, 'task_config.py'))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_logger(
        'mmtune', log_file=osp.join(log_dir, f'{timestamp}.log'))

    logger.info(
        f'Best Hyperparam: \n'
        f'{pformat(ImmutableContainer.decouple(analysis.best_config))}')
    logger.info(
        f'Best Results: \n'
        f'{pformat(ImmutableContainer.decouple(analysis.best_result))}')
    logger.info(f'Best Logdir: {analysis.best_logdir}')
