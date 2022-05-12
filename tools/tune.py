import argparse
import os
from os import path as osp

import mmcv
import ray

from mmtune.apis import log_analysis, tune
from mmtune.mm.tasks import BaseTask, build_task_processor

TASK_NAME = os.getenv('MMTUNE_TASK_NAME')
assert TASK_NAME is not None


def parse_args(task_processor: BaseTask) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='tune')
    parser.add_argument('tune_config', help='tune config file path')
    parser.add_argument(
        '--task-config', default=None, help='taks config file path')
    parser = task_processor.add_arguments(parser)
    parser.add_argument(
        '--address',
        default=None,
        help='the address of the ray cluster to connect to',
    )
    parser.add_argument(
        '--num-cpus',
        default=None,
        help='number of CPUs the user wishes to assign',
    )
    parser.add_argument(
        '--num-gpus',
        default=None,
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
    args = parser.parse_args()
    assert hasattr(args, 'task_config')
    return args


def main():
    task_processor = build_task_processor(TASK_NAME)

    args = parse_args(task_processor)
    tune_config = mmcv.Config.fromfile(args.tune_config)
    task_config = mmcv.Config(
        dict()) if args.task_config is None else mmcv.Config.fromfile(
            args.task_config)
    task_processor.set_base_cfg(task_config)

    file_name = osp.splitext(osp.basename(args.tune_config))[0]
    """
    work_dir is determined in this priority:
    CLI > segment in tune cfg file > segment in task cfg file > tune cfg filename
    """
    args.work_dir = getattr(args, 'work_dir', '') or getattr(
        tune_config, 'work_dir', '') or getattr(
            task_config, 'work_dir', '') or osp.join('./work_dirs', file_name)
    mmcv.mkdir_or_exist(args.work_dir)
    task_processor.set_args(args)
    task_processor.set_rewriters(getattr(tune_config, 'rewriters', []))
    exp_name = args.exp_name or getattr(tune_config, 'exp_name',
                                        '') or file_name

    ray.init(
        address=args.address, num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    assert ray.is_initialized()

    analysis_dir = osp.join(args.work_dir, 'analysis')
    mmcv.mkdir_or_exist(analysis_dir)
    log_analysis(
        tune(task_processor, tune_config, exp_name), tune_config, task_config,
        analysis_dir)


if __name__ == '__main__':
    main()
