# Copyright (c) SI-Analytics. All rights reserved.
import argparse
from argparse import REMAINDER
from os import path as osp

import ray
from mmengine.config.config import Config, DictAction

from siatune.apis import log_analysis
from siatune.tune import Tuner
from siatune.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Tune a model')
    parser.add_argument('config', help='tune config file path')
    parser.add_argument(
        '--work-dir', default=None, help='the dir to save logs and models')
    parser.add_argument(
        '--resume', default=None, help='the experiment path to resume')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
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
        '--trainable-args',
        nargs=REMAINDER,
        type=str,
        help='Rest from the trainable process.',
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # register all modules in siatune into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # resume is determined in this priority: CLI > segment in file
    if args.resume is not None:
        cfg.resume = args.resume

    # task arguments are determined in this priority: CLI > segment in file
    if args.trainable_args is not None:
        cfg.task.args = args.trainable_args

    # set resource
    cfg.task.num_workers = args.num_workers
    cfg.task.num_cpus_per_worker = args.num_cpus_per_worker
    cfg.task.num_gpus_per_worker = args.num_gpus_per_worker

    # init ray
    if not ray.is_initialized():
        ray.init(
            address=args.address,
            num_cpus=args.num_cpus,
            num_gpus=args.num_gpus)
    assert ray.is_initialized()

    # start tuning
    tuner = Tuner.from_cfg(cfg)
    results = tuner.tune()

    log_analysis(
        results, log_dir=osp.join(tuner.work_dir, tuner.experiment_name))
