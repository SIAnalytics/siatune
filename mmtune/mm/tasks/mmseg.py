import argparse
import copy
import os
import time
from os import path as osp
from typing import Optional, Sequence

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmcv.utils import Config, DictAction, get_git_hash

from .builder import TASKS
from .mmtrainbase import MMTrainBasedTask


@TASKS.register_module()
class MMSegmentation(MMTrainBasedTask):
    """MMSegmentation Wrapping class for ray tune."""

    def parse_args(self, args: Sequence[str]) -> argparse.Namespace:
        """Define and parse the necessary arguments for the task.

        Args:
            args (Sequence[str]): The args.
        Returns:
            argparse.Namespace: The parsed args.
        """

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
            help='override some settings in the used config, the key-value '
            'pair in xxx=yyy format will be merged into config file. If the '
            'value to be overwritten is a list, it should be like key="[a,b]" '
            'or key=a,b It also allows nested list/tuple values, e.g. '
            'key="[(a,b),(c,d)]" Note that the quotation marks are necessary '
            'and that no white space is allowed.')
        parser.add_argument(
            '--auto-resume',
            action='store_true',
            help='resume from the latest checkpoint automatically.')
        args = parser.parse_args(args)
        return args

    def build_model(self,
                    cfg: Config,
                    train_cfg: Optional[Config] = None,
                    test_cfg: Optional[Config] = None) -> torch.nn.Module:
        """Build the model from configs.

        Args:
            cfg (Config): The configs.
            train_cfg (Optional[Config]):
                The train opt. Defaults to None.
            test_cfg (Optional[Config]):
                The Test opt. Defaults to None.

        Returns:
            torch.nn.Module: The model.
        """

        from mmseg.models.builder import build_segmentor
        return build_segmentor(cfg, train_cfg, test_cfg)

    def build_dataset(
            self,
            cfg: Config,
            default_args: Optional[Config] = None) -> torch.utils.data.Dataset:
        """Build the dataset from configs.

        Args:
            cfg (Config): The configs.
            default_args (Optional[Config]):
                The default args. Defaults to None.

        Returns:
            torch.utils.data.Dataset: The dataset.
        """

        from mmseg.datasets.builder import build_dataset
        return build_dataset(cfg, default_args)

    def train_model(self,
                    model: torch.nn.Module,
                    dataset: torch.utils.data.Dataset,
                    cfg: Config,
                    distributed: bool = True,
                    validate: bool = False,
                    timestamp: Optional[str] = None,
                    meta: Optional[dict] = None) -> None:
        """Train the model.

        Args:
            model (torch.nn.Module): The model.
            dataset (torch.utils.data.Dataset): The dataset.
            cfg (Config): The configs.
            distributed (bool):
                Whether or not distributed. Defaults to True.
            validate (bool):
                Whether or not validate. Defaults to False.
            timestamp (Optional[str]):
                The timestamp. Defaults to None.
            meta (Optional[dict]): The meta. Defaults to None.
        """

        from mmseg.apis.train import train_segmentor
        return train_segmentor(model, dataset, cfg, distributed, validate,
                               timestamp, meta)

    def run(self, *, args, **kwargs) -> None:
        """Run the task.

        Args:
            args (argparse.Namespace):
                The args that received from context manager.
        """

        from mmseg import __version__
        from mmseg.apis import init_random_seed, set_random_seed
        from mmseg.utils import (collect_env, get_root_logger,
                                 setup_multi_processes)

        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(dist.get_rank())

        cfg = Config.fromfile(args.config)
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        # work_dir is determined in this priority: CLI >
        # segment in file > filename
        if args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            cfg.work_dir = args.work_dir
        elif cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(args.config))[0])
        if args.load_from is not None:
            cfg.load_from = args.load_from
        if args.resume_from is not None:
            cfg.resume_from = args.resume_from

        cfg.auto_resume = args.auto_resume

        # init distributed env first, since logger depends on the dist info.
        distributed = True
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
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        # set multi-process settings
        setup_multi_processes(cfg)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info +  # noqa W504
                    '\n' + dash_line)
        meta['env_info'] = env_info

        # log some basic info
        logger.info(f'Distributed training: {distributed}')
        logger.info(f'Config:\n{cfg.pretty_text}')

        # set random seeds
        seed = init_random_seed(args.seed)
        seed = seed + dist.get_rank() if args.diff_seed else seed
        logger.info(f'Set random seed to {seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(seed, deterministic=args.deterministic)
        cfg.seed = seed
        meta['seed'] = seed
        meta['exp_name'] = osp.basename(args.config)

        model = self.build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        model.init_weights()

        # SyncBN is not support for DP
        logger.info(model)

        datasets = [self.build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(self.build_dataset(val_dataset))
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
        self.train_model(
            model,
            datasets,
            cfg,
            distributed=True,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)
