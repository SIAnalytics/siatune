import time
from os import path as osp

from mmcv.utils import Config, get_logger
from ray import tune

from mmtune.utils import ImmutableContainer


def log_analysis(analysis: tune.ExperimentAnalysis, tune_config: Config,
                 task_config: Config, log_dir: str) -> None:
    tune_config.dump(osp.join(log_dir, 'tune_config.py'))
    task_config.dump(osp.join(log_dir, 'task_config.py'))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_logger(
        'mmtune', log_file=osp.join(log_dir, f'{timestamp}.log'))

    logger.info(
        ('Best Hyperparam', ImmutableContainer.decouple(analysis.best_config)))
    logger.info(
        ('Best Results', ImmutableContainer.decouple(analysis.best_result)))
    logger.info(('Best Logdir', analysis.best_logdir))
    logger.info(analysis.results)
