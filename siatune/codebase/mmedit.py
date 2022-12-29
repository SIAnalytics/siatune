# Copyright (c) SI-Analytics. All rights reserved.
import argparse
import copy
import logging
import os
import time
from os import path as osp
from typing import Sequence

from mmengine.config import Config, DictAction

from siatune.version import IS_DEPRECATED_MMCV
from .builder import TASKS
from .mm import MMBaseTask

if IS_DEPRECATED_MMCV:

    @TASKS.register_module()
    class MMEditing(MMBaseTask):
        """MMEditing wrapper class for `ray.tune`.

        It is modified from https://github.com/open-mmlab/mmediting/blob/v0.15.0/tools/train.py

        Args:
            args (argparse.Namespace): The arguments for `tools/train.py`
                script file. It is parsed by :method:`parse_args`.
            num_workers (int): The number of workers to launch.
            num_cpus_per_worker (int): The number of CPUs per worker.
                Default to 1.
            num_gpus_per_worker (int): The number of GPUs per worker.
                Since it must be equal to `num_workers` attribute, it
                is not used in MMEditing.
            rewriters (list[dict] | dict, optional): Context redefinition
                pipeline. Default to None.
        """

        VERSION = 'v0.15.0'

        def parse_args(self, task_args: Sequence[str]):
            from mmcv import DictAction

            parser = argparse.ArgumentParser(description='Train an editor')
            parser.add_argument('config', help='train config file path')
            parser.add_argument(
                '--work-dir', help='the dir to save logs and models')
            parser.add_argument(
                '--resume-from', help='the checkpoint file to resume from')
            parser.add_argument(
                '--no-validate',
                action='store_true',
                help='whether not to evaluate the checkpoint during training')
            parser.add_argument(
                '--gpus',
                type=int,
                default=1,
                help='number of gpus to use '
                '(only applicable to non-distributed training)')
            parser.add_argument(
                '--seed', type=int, default=None, help='random seed')
            parser.add_argument(
                '--diff_seed',
                action='store_true',
                help='Whether or not set different seeds for different ranks')
            parser.add_argument(
                '--deterministic',
                action='store_true',
                help='whether to set deterministic options for CUDNN backend.')
            parser.add_argument(
                '--cfg-options',
                nargs='+',
                action=DictAction,
                help=
                'override some settings in the used config, the key-value pair '
                'in xxx=yyy format will be merged into config file. If the value to '
                'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                'Note that the quotation marks are necessary and that no white space '
                'is allowed.')
            parser.add_argument(
                '--launcher',
                choices=['none', 'pytorch', 'slurm', 'mpi'],
                default='none',
                help='job launcher')
            parser.add_argument('--local_rank', type=int, default=0)
            parser.add_argument(
                '--autoscale-lr',
                action='store_true',
                help='automatically scale lr with the number of gpus')
            args = parser.parse_args(task_args)
            if 'LOCAL_RANK' not in os.environ:
                os.environ['LOCAL_RANK'] = str(args.local_rank)

            return args

        def run(self, args: argparse.Namespace):
            """Run the task.

            Args:
                args (argparse.Namespace):
                    The args that received from context manager.
            """

            import mmcv
            import torch
            import torch.distributed as dist
            from mmcv import Config
            from mmcv.runner import init_dist
            from mmedit import __version__
            from mmedit.apis import (init_random_seed, set_random_seed,
                                     train_model)
            from mmedit.datasets import build_dataset
            from mmedit.models import build_model
            from mmedit.utils import (collect_env, get_root_logger,
                                      setup_multi_processes)

            cfg = Config.fromfile(args.config)

            if args.cfg_options is not None:
                cfg.merge_from_dict(args.cfg_options)

            # set multi-process settings
            setup_multi_processes(cfg)

            # set cudnn_benchmark
            if cfg.get('cudnn_benchmark', False):
                torch.backends.cudnn.benchmark = True
            # update configs according to CLI args
            if args.work_dir is not None:
                cfg.work_dir = args.work_dir
            if args.resume_from is not None:
                cfg.resume_from = args.resume_from
            cfg.gpus = args.gpus

            if args.autoscale_lr:
                # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
                cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

            # init distributed env first, since logger depends on the dist info.
            if args.launcher == 'none':
                distributed = False
            else:
                distributed = True
                init_dist(args.launcher, **cfg.dist_params)

            # create work_dir
            mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
            # init the logger before other steps
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
            logger = get_root_logger(
                log_file=log_file, log_level=cfg.log_level)

            # log env info
            env_info_dict = collect_env.collect_env()
            env_info = '\n'.join(
                [f'{k}: {v}' for k, v in env_info_dict.items()])
            dash_line = '-' * 60 + '\n'
            logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                        dash_line)

            # log some basic info
            logger.info('Distributed training: {}'.format(distributed))
            logger.info('mmedit Version: {}'.format(__version__))
            logger.info('Config:\n{}'.format(cfg.text))

            # set random seeds
            seed = init_random_seed(args.seed)
            seed = seed + dist.get_rank() if args.diff_seed else seed
            logger.info('Set random seed to {}, deterministic: {}'.format(
                seed, args.deterministic))
            set_random_seed(seed, deterministic=args.deterministic)
            cfg.seed = seed

            model = build_model(
                cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

            datasets = [build_dataset(cfg.data.train)]
            if len(cfg.workflow) == 2:
                val_dataset = copy.deepcopy(cfg.data.val)
                val_dataset.pipeline = cfg.data.train.pipeline
                datasets.append(build_dataset(val_dataset))
            if cfg.checkpoint_config is not None:
                # save version, config file content and class names in
                # checkpoints as meta data
                cfg.checkpoint_config.meta = dict(
                    mmedit_version=__version__,
                    config=cfg.text,
                )

            # meta information
            meta = dict()
            if cfg.get('exp_name', None) is None:
                cfg['exp_name'] = osp.splitext(osp.basename(cfg.work_dir))[0]
            meta['exp_name'] = cfg.exp_name
            meta['mmedit Version'] = __version__
            meta['seed'] = seed
            meta['env_info'] = env_info

            # add an attribute for visualization convenience
            train_model(
                model,
                datasets,
                cfg,
                distributed=distributed,
                validate=(not args.no_validate),
                timestamp=timestamp,
                meta=meta)

else:

    @TASKS.register_module()
    class MMEditing(MMBaseTask):
        """MMEditing wrapper class for `ray.tune`.

        It is modified from https://github.com/open-mmlab/mmediting/blob/v1.0.0rc4/tools/train.py

        Args:
            args (argparse.Namespace): The arguments for `tools/train.py`
                script file. It is parsed by :method:`parse_args`.
            num_workers (int): The number of workers to launch.
            num_cpus_per_worker (int): The number of CPUs per worker.
                Default to 1.
            num_gpus_per_worker (int): The number of GPUs per worker.
                Since it must be equal to `num_workers` attribute, it
                is not used in MMEditing.
            rewriters (list[dict] | dict, optional): Context redefinition
                pipeline. Default to None.
        """

        VERSION = 'v1.0.0rc4'

        def parse_args(self, task_args: Sequence[str]):
            parser = argparse.ArgumentParser(description='Train a model')
            parser.add_argument('config', help='train config file path')
            parser.add_argument(
                '--work-dir', help='the dir to save logs and models')
            parser.add_argument(
                '--resume',
                action='store_true',
                help='Whether to resume checkpoint.')
            parser.add_argument(
                '--amp',
                action='store_true',
                default=False,
                help='enable automatic-mixed-precision training')
            parser.add_argument(
                '--auto-scale-lr',
                action='store_true',
                help='enable automatically scaling LR.')
            parser.add_argument(
                '--cfg-options',
                nargs='+',
                action=DictAction,
                help=
                'override some settings in the used config, the key-value pair '
                'in xxx=yyy format will be merged into config file. If the value to '
                'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                'Note that the quotation marks are necessary and that no white space '
                'is allowed.')
            parser.add_argument(
                '--launcher',
                choices=['none', 'pytorch', 'slurm', 'mpi'],
                default='none',
                help='job launcher')
            parser.add_argument('--local_rank', type=int, default=0)
            args = parser.parse_args(task_args)
            if 'LOCAL_RANK' not in os.environ:
                os.environ['LOCAL_RANK'] = str(args.local_rank)

            return args

        def run(self, args: argparse.Namespace):
            """Run the task.

            Args:
                args (argparse.Namespace):
                    The args that received from context manager.
            """
            from mmedit.utils import print_colored_log, register_all_modules
            from mmengine.runner import Runner

            # register all modules in mmedit into the registries
            # do not init the default scope here because it will be init in the runner
            register_all_modules(init_default_scope=False)

            # load config
            cfg = Config.fromfile(args.config)
            cfg.launcher = args.launcher
            if args.cfg_options is not None:
                cfg.merge_from_dict(args.cfg_options)

            # work_dir is determined in this priority: CLI > segment in file > filename
            if args.work_dir:  # none or empty str
                # update configs according to CLI args if args.work_dir is not None
                cfg.work_dir = args.work_dir
            elif cfg.get('work_dir', None) is None:
                # use config filename as default work_dir if cfg.work_dir is None
                cfg.work_dir = osp.join(
                    './work_dirs',
                    osp.splitext(osp.basename(args.config))[0])

            # enable automatic-mixed-precision training
            if args.amp is True:
                if ('constructor' not in cfg.optim_wrapper) or \
                        cfg.optim_wrapper['constructor'] == 'DefaultOptimWrapperConstructor': # noqa
                    optim_wrapper = cfg.optim_wrapper.type
                    if optim_wrapper == 'AmpOptimWrapper':
                        print_colored_log(
                            'AMP training is already enabled in your config.',
                            logger='current',
                            level=logging.WARNING)
                    else:
                        assert optim_wrapper == 'OptimWrapper', (
                            '`--amp` is only supported when the optimizer wrapper '
                            f'`type is OptimWrapper` but got {optim_wrapper}.')
                        cfg.optim_wrapper.type = 'AmpOptimWrapper'
                        cfg.optim_wrapper.loss_scale = 'dynamic'
                else:
                    for key, val in cfg.optim_wrapper.items():
                        if isinstance(val, dict) and 'type' in val:
                            assert val.type == 'OptimWrapper', (
                                '`--amp` is only supported when the optimizer wrapper '
                                f'`type is OptimWrapper` but got {val.type}.')
                            cfg.optim_wrapper[key].type = 'AmpOptimWrapper'
                            cfg.optim_wrapper[key].loss_scale = 'dynamic'

            if args.resume:
                cfg.resume = True

            # build the runner from config
            runner = Runner.from_cfg(cfg)

            print_colored_log(f'Working directory: {cfg.work_dir}')
            print_colored_log(f'Log directiry: {runner._log_dir}')

            # start training
            runner.train()

            print_colored_log(f'Log saved under {runner._log_dir}')
            print_colored_log(f'Checkpoint saved under {cfg.work_dir}')
