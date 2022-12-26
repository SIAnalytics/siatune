# Copyright (c) SI-Analytics. All rights reserved.
import os
import shutil
from os import path as osp
from pprint import pformat
from typing import Optional

import mmengine
from ray.tune import ResultGrid

from siatune.utils import ImmutableContainer
from siatune.utils.logger import get_root_logger


def log_analysis(results: ResultGrid, log_dir: Optional[str] = None) -> None:
    """Log the analysis of the experiment.

    Args:
        results (ResultGrid): Experiment results of `Tuner.tune()`.
        log_dir (str, optional): The log dir. Defaults to None.
    """

    log_dir = osp.join(log_dir or os.getcwd(), 'best_trial')
    mmengine.mkdir_or_exist(log_dir)

    logger = get_root_logger(log_file=osp.join(log_dir, 'best_trial.log'))
    result = results.get_best_result()

    logger.info(f'Best Logdir: {result.log_dir}')
    logger.info(f'Best Result: \n'
                f'{pformat(ImmutableContainer.decouple(result))}')
    logger.info(f'Best Hyperparam: \n'
                f'{pformat(ImmutableContainer.decouple(result.config))}')

    shutil.copytree(result.log_dir, osp.join(log_dir, 'log'))
