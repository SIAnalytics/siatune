# Copyright (c) SI-Analytics. All rights reserved.
import argparse
import os
import subprocess
import time
from glob import glob
from os import path as osp
from typing import List, Optional

from mmcv import Config

from ..blackbox import BlackBoxTask
from ..builder import TASKS
from ._bystander import reporter_factory
from ._utils import get_installed_path, module_full_name, revert_args


@TASKS.register_module()
class BystanderTrainBasedTask(BlackBoxTask):

    def __init__(self,
                 pkg_name: str,
                 metric: str,
                 rewriters: List[dict] = []) -> None:
        """Initialize the task.

        Args:
            pkg_name (str):
            metric (str):
            rewriters (List[dict]):
                Context redefinition pipeline. Defaults to [].
        """
        self._train_script: str = self._get_train_script(pkg_name)
        self._metric: str = metric
        self._args: Optional[argparse.Namespace] = None
        self._rewriters: List[dict] = rewriters

    def _get_train_script(self, pkg_name: str) -> str:
        pkg_full_name = module_full_name(pkg_name)
        if pkg_full_name == '':
            raise ValueError("Can't determine a unique "
                             f'package given abbreviation {pkg_name}')
        pkg_root = get_installed_path(pkg_full_name)
        train_script = osp.join(pkg_root, '.mim', 'tools', 'train.py')
        if not osp.exists(train_script):
            _alternative = osp.join(pkg_root, 'tools', 'train.py')
            if osp.exists(_alternative):
                train_script = _alternative
            else:
                raise ValueError("Can't find train script")
        return train_script

    def _get_work_dir(self, args: argparse.Namespace) -> str:
        work_dir = getattr(args, 'work_dir', '')
        if work_dir:
            return work_dir
        if not hasattr(args, 'config'):
            raise ValueError
        cfg = Config.fromfile(args.config)
        work_dir = cfg.get('work_dir', '')
        if work_dir:
            return work_dir
        raise ValueError

    def _get_deffered_log(self,
                          work_dir: str,
                          log_ext: str = '.log',
                          timeout: float = 60 * 10) -> str:
        t_bound = time.time() + timeout
        files: List[str]
        while True:
            files = glob(osp.join(work_dir, '*.', log_ext))
            if files:
                break
            elif time.time() > t_bound:
                raise RuntimeError
        if len(files) > 1:
            raise ValueError
        return files.pop()

    def run(self, *, args: argparse.Namespace, **kwargs) -> None:
        port: int = kwargs.get('port', 29500)
        cmd: List[str]
        if args.launcher == 'none':
            cmd = ['python', self._train_script]
        elif args.launcher == 'pytorch':
            cmd = [
                'python',
                '-m',
                'torch.distributed.launch',
                f'--nproc_per_node={self.num_workers}',
                f'--master_port={port}',
                self._train_script,
            ]
        else:
            NotImplementedError
        cmd.extend(revert_args(args))

        worker = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            env=dict(os.environ, MASTER_PORT=str(port)))
        work_dir = self._get_work_dir(args)
        log_file = self._get_deffered_log(work_dir)
        reporter = reporter_factory(log_file, self._metric)
        reporter.start()

        _, err = worker.communicate()
        if worker.returncode != 0:
            raise Exception(err)
        else:
            reporter.shutdown()
        return
