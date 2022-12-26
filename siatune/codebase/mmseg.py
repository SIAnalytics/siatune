# Copyright (c) SI-Analytics. All rights reserved.
import argparse
import copy
import logging
import os
import time
import warnings
from os import path as osp
from typing import Sequence

from mmengine.config import Config, DictAction

from siatune.version import IS_DEPRECATED_MMCV
from .builder import TASKS
from .mm import MMBaseTask

if IS_DEPRECATED_MMCV:

    @TASKS.register_module()
    class MMSegmentation(MMBaseTask):
        """MMSegmentation wrapper class for `ray.tune`.

        It is modified from https://github.com/open-mmlab/mmsegmentation/blob/v0.25.0/tools/train.py

        Args:
            args (argparse.Namespace): The arguments for `tools/train.py`
                script file. It is parsed by :method:`parse_args`.
            num_workers (int): The number of workers to launch.
            num_cpus_per_worker (int): The number of CPUs per worker.
                Default to 1.
            num_gpus_per_worker (int): The number of GPUs per worker.
                Since it must be equal to `num_workers` attribute, it
                is not used in MMSegmentation.
            rewriters (list[dict] | dict, optional): Context redefinition
                pipeline. Default to None.
        """

        VERSION = 'v0.25.0'

        def parse_args(self, task_args: Sequence[str]):
            parser = argparse.ArgumentParser(description='Train a segmentor')
            parser.add_argument('config', help='train config file path')
            parser.add_argument(
                '--work-dir', help='the dir to save logs and models')
            parser.add_argument(
                '--load-from', help='the checkpoint file to load weights from')
            parser.add_argument(
                '--resume-from', help='the checkpoint file to resume from')
            parser.add_argument(
                '--no-validate',
                action='store_true',
                help='whether not to evaluate the checkpoint during training')
            group_gpus = parser.add_mutually_exclusive_group()
            group_gpus.add_argument(
                '--gpus',
                type=int,
                help='(Deprecated, please use --gpu-id) number of gpus to use '
                '(only applicable to non-distributed training)')
            group_gpus.add_argument(
                '--gpu-ids',
                type=int,
                nargs='+',
                help='(Deprecated, please use --gpu-id) ids of gpus to use '
                '(only applicable to non-distributed training)')
            group_gpus.add_argument(
                '--gpu-id',
                type=int,
                default=0,
                help='id of gpu to use '
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
                '--options',
                nargs='+',
                action=DictAction,
                help=
                "--options is deprecated in favor of --cfg_options' and it will "
                'not be supported in version v0.22.0. Override some settings in the '
                'used config, the key-value pair in xxx=yyy format will be merged '
                'into config file. If the value to be overwritten is a list, it '
                'should be like key="[a,b]" or key=a,b It also allows nested '
                'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
                'marks are necessary and that no white space is allowed.')
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
                '--auto-resume',
                action='store_true',
                help='resume from the latest checkpoint automatically.')
            args = parser.parse_args(task_args)
            if 'LOCAL_RANK' not in os.environ:
                os.environ['LOCAL_RANK'] = str(args.local_rank)

            if args.options and args.cfg_options:
                raise ValueError(
                    '--options and --cfg-options cannot be both '
                    'specified, --options is deprecated in favor of --cfg-options. '
                    '--options will not be supported in version v0.22.0.')
            if args.options:
                warnings.warn(
                    '--options is deprecated in favor of --cfg-options. '
                    '--options will not be supported in version v0.22.0.')
                args.cfg_options = args.options

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
            from mmcv.cnn.utils import revert_sync_batchnorm
            from mmcv.runner import get_dist_info, init_dist
            from mmcv.utils import Config, get_git_hash
            from mmseg import __version__
            from mmseg.apis import (init_random_seed, set_random_seed,
                                    train_segmentor)
            from mmseg.datasets import build_dataset
            from mmseg.models import build_segmentor
            from mmseg.utils import (collect_env, get_device, get_root_logger,
                                     setup_multi_processes)

            cfg = Config.fromfile(args.config)
            if args.cfg_options is not None:
                cfg.merge_from_dict(args.cfg_options)

            # set cudnn_benchmark
            if cfg.get('cudnn_benchmark', False):
                torch.backends.cudnn.benchmark = True

            # work_dir is determined in this priority: CLI > segment in file > filename
            if args.work_dir is not None:
                # update configs according to CLI args if args.work_dir is not None
                cfg.work_dir = args.work_dir
            elif cfg.get('work_dir', None) is None:
                # use config filename as default work_dir if cfg.work_dir is None
                cfg.work_dir = osp.join(
                    './work_dirs',
                    osp.splitext(osp.basename(args.config))[0])
            if args.load_from is not None:
                cfg.load_from = args.load_from
            if args.resume_from is not None:
                cfg.resume_from = args.resume_from
            if args.gpus is not None:
                cfg.gpu_ids = range(1)
                warnings.warn('`--gpus` is deprecated because we only support '
                              'single GPU mode in non-distributed training. '
                              'Use `gpus=1` now.')
            if args.gpu_ids is not None:
                cfg.gpu_ids = args.gpu_ids[0:1]
                warnings.warn(
                    '`--gpu-ids` is deprecated, please use `--gpu-id`. '
                    'Because we only support single GPU mode in '
                    'non-distributed training. Use the first GPU '
                    'in `gpu_ids` now.')
            if args.gpus is None and args.gpu_ids is None:
                cfg.gpu_ids = [args.gpu_id]

            cfg.auto_resume = args.auto_resume

            # init distributed env first, since logger depends on the dist info.
            if args.launcher == 'none':
                distributed = False
            else:
                distributed = True
                init_dist(args.launcher, **cfg.dist_params)
                # gpu_ids is used to calculate iter when resuming checkpoint
                _, world_size = get_dist_info()
                cfg.gpu_ids = range(world_size)

            # create work_dir
            mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
            # dump config
            cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
            # init the logger before other steps
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
            logger = get_root_logger(
                log_file=log_file, log_level=cfg.log_level)

            # set multi-process settings
            setup_multi_processes(cfg)

            # init the meta dict to record some important information such as
            # environment info and seed, which will be logged
            meta = dict()
            # log env info
            env_info_dict = collect_env()
            env_info = '\n'.join(
                [f'{k}: {v}' for k, v in env_info_dict.items()])
            dash_line = '-' * 60 + '\n'
            logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                        dash_line)
            meta['env_info'] = env_info

            # log some basic info
            logger.info(f'Distributed training: {distributed}')
            logger.info(f'Config:\n{cfg.pretty_text}')

            # set random seeds
            cfg.device = get_device()
            seed = init_random_seed(args.seed, device=cfg.device)
            seed = seed + dist.get_rank() if args.diff_seed else seed
            logger.info(f'Set random seed to {seed}, '
                        f'deterministic: {args.deterministic}')
            set_random_seed(seed, deterministic=args.deterministic)
            cfg.seed = seed
            meta['seed'] = seed
            meta['exp_name'] = osp.basename(args.config)

            model = build_segmentor(
                cfg.model,
                train_cfg=cfg.get('train_cfg'),
                test_cfg=cfg.get('test_cfg'))
            model.init_weights()

            # SyncBN is not support for DP
            if not distributed:
                warnings.warn(
                    'SyncBN is only supported with DDP. To be compatible with DP, '
                    'we convert SyncBN to BN. Please use dist_train.sh which can '
                    'avoid this error.')
                model = revert_sync_batchnorm(model)

            logger.info(model)

            datasets = [build_dataset(cfg.data.train)]
            if len(cfg.workflow) == 2:
                val_dataset = copy.deepcopy(cfg.data.val)
                val_dataset.pipeline = cfg.data.train.pipeline
                datasets.append(build_dataset(val_dataset))
            if cfg.checkpoint_config is not None:
                # save mmseg version, config file content and class names in
                # checkpoints as meta data
                cfg.checkpoint_config.meta = dict(
                    mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
                    config=cfg.pretty_text,
                    CLASSES=datasets[0].CLASSES,
                    PALETTE=datasets[0].PALETTE)
            # add an attribute for visualization convenience
            model.CLASSES = datasets[0].CLASSES
            # passing checkpoint meta for saving best checkpoint
            meta.update(cfg.checkpoint_config.meta)
            train_segmentor(
                model,
                datasets,
                cfg,
                distributed=distributed,
                validate=(not args.no_validate),
                timestamp=timestamp,
                meta=meta)

else:

    @TASKS.register_module()
    class MMSegmentation(MMBaseTask):
        """MMSegmentation wrapper class for `ray.tune`.

        It is modified from https://github.com/open-mmlab/mmsegmentation/tree/v1.0.0rc2/tools/train.py

        Args:
            args (argparse.Namespace): The arguments for `tools/train.py`
                script file. It is parsed by :method:`parse_args`.
            num_workers (int): The number of workers to launch.
            num_cpus_per_worker (int): The number of CPUs per worker.
                Default to 1.
            num_gpus_per_worker (int): The number of GPUs per worker.
                Since it must be equal to `num_workers` attribute, it
                is not used in MMSegmentation.
            rewriters (list[dict] | dict, optional): Context redefinition
                pipeline. Default to None.
        """

        VERSION = 'v1.0.0rc2'

        def parse_args(self, task_args: Sequence[str]):
            parser = argparse.ArgumentParser(description='Train a segmentor')
            parser.add_argument('config', help='train config file path')
            parser.add_argument(
                '--work-dir', help='the dir to save logs and models')
            parser.add_argument(
                '--resume',
                action='store_true',
                default=False,
                help=
                'resume from the latest checkpoint in the work_dir automatically'
            )
            parser.add_argument(
                '--amp',
                action='store_true',
                default=False,
                help='enable automatic-mixed-precision training')
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
            from mmengine.logging import print_log
            from mmengine.registry import RUNNERS
            from mmengine.runner import Runner
            from mmseg.utils import register_all_modules

            # register all modules in mmseg into the registries
            # do not init the default scope here because it will be init in the runner
            register_all_modules(init_default_scope=False)

            # load config
            cfg = Config.fromfile(args.config)
            cfg.launcher = args.launcher
            if args.cfg_options is not None:
                cfg.merge_from_dict(args.cfg_options)

            # work_dir is determined in this priority: CLI > segment in file > filename
            if args.work_dir is not None:
                # update configs according to CLI args if args.work_dir is not None
                cfg.work_dir = args.work_dir
            elif cfg.get('work_dir', None) is None:
                # use config filename as default work_dir if cfg.work_dir is None
                cfg.work_dir = osp.join(
                    './work_dirs',
                    osp.splitext(osp.basename(args.config))[0])

            # enable automatic-mixed-precision training
            if args.amp is True:
                optim_wrapper = cfg.optim_wrapper.type
                if optim_wrapper == 'AmpOptimWrapper':
                    print_log(
                        'AMP training is already enabled in your config.',
                        logger='current',
                        level=logging.WARNING)
                else:
                    assert optim_wrapper == 'OptimWrapper', (
                        '`--amp` is only supported when the optimizer wrapper type is '
                        f'`OptimWrapper` but got {optim_wrapper}.')
                    cfg.optim_wrapper.type = 'AmpOptimWrapper'
                    cfg.optim_wrapper.loss_scale = 'dynamic'

            # resume training
            cfg.resume = args.resume

            # build the runner from config
            if 'runner_type' not in cfg:
                # build the default runner
                runner = Runner.from_cfg(cfg)
            else:
                # build customized runner from the registry
                # if 'runner_type' is set in the cfg
                runner = RUNNERS.build(cfg)

            # start training
            runner.train()
