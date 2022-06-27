from argparse import REMAINDER, ArgumentParser, Namespace
from os import path as osp

import mmcv
import ray
from mmcv import Config

from mmtune.apis import log_analysis, tune
from mmtune.mm.tasks import build_task_processor


def parse_args() -> Namespace:
    """Parse arguments.

    Returns:
        Namespace: The parsed arguments.
    """

    parser = ArgumentParser(description='tune')
    parser.add_argument('tune_config', help='tune config file path')
    parser.add_argument(
        '--work-dir', default=None, help='the dir to save logs and models')
    parser.add_argument(
        '--address',
        default=None,
        help='the address of the ray cluster to connect to',
    )
    parser.add_argument(
        '--num-cpus',
        default=None,
        type=int,
        help='number of CPUs the user wishes to assign',
    )
    parser.add_argument(
        '--num-gpus',
        default=None,
        type=int,
        help='number of GPUs the user wishes to assign',
    )
    parser.add_argument(
        '--exp-name', type=str, help='name of experiment', default='')
    parser.add_argument(
        '--num-workers', type=int, default=1, help='number of workers to use')
    parser.add_argument(
        '--num-gpus-per-worker',
        type=int,
        default=1,
        help='number of gpus each worker uses.',
    )
    parser.add_argument(
        '--num-cpus-per-worker',
        type=int,
        default=1,
        help='number of gpus each worker uses.',
    )
    parser.add_argument(
        'trainable_args',
        nargs=REMAINDER,
        type=str,
        help='Rest from the trainable process.',
    )
    args = parser.parse_args()
    return args


def main() -> None:
    """Main function."""

    args = parse_args()
    tune_config = Config.fromfile(args.tune_config)

    task_processor = build_task_processor(tune_config.task)
    task_processor.set_args(args.trainable_args)

    file_name = osp.splitext(osp.basename(args.tune_config))[0]
    exp_name = args.exp_name or tune_config.get('exp_name', file_name)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        tune_config.work_dir = args.work_dir
    elif tune_config.get('work_dir', None) is None:
        tune_config.work_dir = osp.join('./work_dirs', file_name)
    mmcv.mkdir_or_exist(tune_config.work_dir)
    # work_dir in task is overridden with work_dir in tune
    if hasattr(task_processor.args, 'work_dir'):
        task_processor.args.work_dir = tune_config.work_dir

    ray.init(
        address=args.address, num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    assert ray.is_initialized()

    task_config = getattr(task_processor.args, 'config', None)
    if task_config is not None:
        task_config = Config.fromfile(task_config)

    analysis_dir = osp.join(tune_config.work_dir, 'analysis')
    mmcv.mkdir_or_exist(analysis_dir)
    log_analysis(
        tune(task_processor, tune_config, exp_name),
        tune_config,
        task_config=task_config,
        log_dir=analysis_dir)
