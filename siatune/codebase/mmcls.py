# Copyright (c) SI-Analytics. All rights reserved.
import argparse
import copy
import os
import time
import warnings
from os import path as osp
from typing import Sequence

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from siatune.version import IS_DEPRECATED_MMCV
from .builder import TASKS
from .mm import MMBaseTask

if IS_DEPRECATED_MMCV:

    @TASKS.register_module()
    class MMClassification(MMBaseTask):
        """MMClassification wrapper class for `ray.tune`.

        It is modified from https://github.com/open-mmlab/mmclassification/blob/v0.23.2/tools/train.py

        Args:
            args (argparse.Namespace): The arguments for `tools/train.py`
                script file. It is parsed by :method:`parse_args`.
            num_workers (int): The number of workers to launch.
            num_cpus_per_worker (int): The number of CPUs per worker.
                Default to 1.
            num_gpus_per_worker (int): The number of GPUs per worker.
                Since it must be equal `num_workers` attribute, it is
                not used in MMClassification.
            rewriters (list[dict] | dict, optional): Context redefinition
                pipeline. Default to None.
        """

        VERSION = 'v0.23.2'

        def parse_args(self, task_args: Sequence[str]):
            parser = argparse.ArgumentParser(description='Train a model')
            parser.add_argument('config', help='train config file path')
            parser.add_argument(
                '--work-dir', help='the dir to save logs and models')
            parser.add_argument(
                '--resume-from', help='the checkpoint file to resume from')
            parser.add_argument(
                '--no-validate',
                action='store_true',
                help='whether not to evaluate the checkpoint during training')
            group_gpus = parser.add_mutually_exclusive_group()
            group_gpus.add_argument(
                '--device', help='device used for training. (Deprecated)')
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
                '--ipu-replicas',
                type=int,
                default=None,
                help='num of ipu replicas to use')
            parser.add_argument(
                '--seed', type=int, default=None, help='random seed')
            parser.add_argument(
                '--diff-seed',
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
            from mmcls import __version__
            from mmcls.apis import (init_random_seed, set_random_seed,
                                    train_model)
            from mmcls.datasets import build_dataset
            from mmcls.models import build_classifier
            from mmcls.utils import (auto_select_device, collect_env,
                                     get_root_logger, setup_multi_processes)
            from mmcv import Config
            from mmcv.runner import get_dist_info, init_dist

            cfg = Config.fromfile(args.config)
            if args.cfg_options is not None:
                cfg.merge_from_dict(args.cfg_options)

            # set multi-process settings
            setup_multi_processes(cfg)

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

            if args.ipu_replicas is not None:
                cfg.ipu_replicas = args.ipu_replicas
                args.device = 'ipu'

            # init distributed env first, since logger depends on the dist info.
            if args.launcher == 'none':
                distributed = False
            else:
                distributed = True
                init_dist(args.launcher, **cfg.dist_params)
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

            # init the meta dict to record some important information such as
            # environment info and seed, which will be logged
            meta = dict()
            # log env info
            env_info_dict = collect_env()
            env_info = '\n'.join([(f'{k}: {v}')
                                  for k, v in env_info_dict.items()])
            dash_line = '-' * 60 + '\n'
            logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                        dash_line)
            meta['env_info'] = env_info

            # log some basic info
            logger.info(f'Distributed training: {distributed}')
            logger.info(f'Config:\n{cfg.pretty_text}')

            # set random seeds
            cfg.device = args.device or auto_select_device()
            seed = init_random_seed(args.seed, device=cfg.device)
            seed = seed + dist.get_rank() if args.diff_seed else seed
            logger.info(f'Set random seed to {seed}, '
                        f'deterministic: {args.deterministic}')
            set_random_seed(seed, deterministic=args.deterministic)
            cfg.seed = seed
            meta['seed'] = seed

            model = build_classifier(cfg.model)
            model.init_weights()

            datasets = [build_dataset(cfg.data.train)]
            if len(cfg.workflow) == 2:
                val_dataset = copy.deepcopy(cfg.data.val)
                val_dataset.pipeline = cfg.data.train.pipeline
                datasets.append(build_dataset(val_dataset))

            # save mmcls version, config file content and class names in
            # runner as meta data
            meta.update(
                dict(
                    mmcls_version=__version__,
                    config=cfg.pretty_text,
                    CLASSES=datasets[0].CLASSES))

            # add an attribute for visualization convenience
            train_model(
                model,
                datasets,
                cfg,
                distributed=distributed,
                validate=(not args.no_validate),
                timestamp=timestamp,
                device=cfg.device,
                meta=meta)
else:

    @TASKS.register_module()
    class MMClassification(MMBaseTask):
        """MMClassification wrapper class for `ray.tune`.

        It is modified from https://github.com/open-mmlab/mmclassification/blob/v1.0.0rc4/tools/train.py

        Args:
            args (argparse.Namespace): The arguments for `tools/train.py`
                script file. It is parsed by :method:`parse_args`.
            num_workers (int): The number of workers to launch.
            num_cpus_per_worker (int): The number of CPUs per worker.
                Default to 1.
            num_gpus_per_worker (int): The number of GPUs per worker.
                Since it must be equal `num_workers` attribute, it is
                not used in MMClassification.
        """

        VERSION = 'v1.0.0rc4'

        def parse_args(self, task_args: Sequence[str]):
            parser = argparse.ArgumentParser(description='Train a classifier')
            parser.add_argument('config', help='train config file path')
            parser.add_argument(
                '--work-dir', help='the dir to save logs and models')
            parser.add_argument(
                '--resume',
                nargs='?',
                type=str,
                const='auto',
                help='If specify checkpoint path, resume from it, while if not '
                'specify, try to auto resume from the latest checkpoint '
                'in the work directory.')
            parser.add_argument(
                '--amp',
                action='store_true',
                help='enable automatic-mixed-precision training')
            parser.add_argument(
                '--no-validate',
                action='store_true',
                help='whether not to evaluate the checkpoint during training')
            parser.add_argument(
                '--auto-scale-lr',
                action='store_true',
                help='whether to auto scale the learning rate according to the '
                'actual batch size and the original batch size.')
            parser.add_argument(
                '--no-pin-memory',
                action='store_true',
                help='whether to disable the pin_memory option in dataloaders.'
            )
            parser.add_argument(
                '--no-persistent-workers',
                action='store_true',
                help=
                'whether to disable the persistent_workers option in dataloaders.'
            )
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
            from copy import deepcopy

            from mmcls.utils import register_all_modules
            from mmengine.runner import Runner

            def merge_args(cfg, args):
                """Merge CLI arguments to config."""
                if args.no_validate:
                    cfg.val_cfg = None
                    cfg.val_dataloader = None
                    cfg.val_evaluator = None

                cfg.launcher = args.launcher

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
                    optim_wrapper = cfg.optim_wrapper.get(
                        'type', 'OptimWrapper')
                    assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], \
                        '`--amp` is not supported custom optimizer wrapper type ' \
                        f'`{optim_wrapper}.'
                    cfg.optim_wrapper.type = 'AmpOptimWrapper'
                    cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

                # resume training
                if args.resume == 'auto':
                    cfg.resume = True
                    cfg.load_from = None
                elif args.resume is not None:
                    cfg.resume = True
                    cfg.load_from = args.resume

                # enable auto scale learning rate
                if args.auto_scale_lr:
                    cfg.auto_scale_lr.enable = True

                # set dataloader args
                default_dataloader_cfg = ConfigDict(
                    pin_memory=True,
                    persistent_workers=True,
                    collate_fn=dict(type='default_collate'),
                )
                if digit_version(TORCH_VERSION) < digit_version('1.8.0'):
                    default_dataloader_cfg.persistent_workers = False

                def set_default_dataloader_cfg(cfg, field):
                    if cfg.get(field, None) is None:
                        return
                    dataloader_cfg = deepcopy(default_dataloader_cfg)
                    dataloader_cfg.update(cfg[field])
                    cfg[field] = dataloader_cfg
                    if args.no_pin_memory:
                        cfg[field]['pin_memory'] = False
                    if args.no_persistent_workers:
                        cfg[field]['persistent_workers'] = False

                set_default_dataloader_cfg(cfg, 'train_dataloader')
                set_default_dataloader_cfg(cfg, 'val_dataloader')
                set_default_dataloader_cfg(cfg, 'test_dataloader')

                if args.cfg_options is not None:
                    cfg.merge_from_dict(args.cfg_options)

                return cfg

            # register all modules in mmcls into the registries
            # do not init the default scope here because it will be init in the runner
            register_all_modules(init_default_scope=False)

            # load config
            cfg = Config.fromfile(args.config)

            # merge cli arguments to config
            cfg = merge_args(cfg, args)

            # build the runner from config
            runner = Runner.from_cfg(cfg)

            # start training
            runner.train()
