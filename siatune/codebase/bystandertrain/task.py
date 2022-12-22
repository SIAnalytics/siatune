# Copyright (c) SI-Analytics. All rights reserved.
import argparse
import multiprocessing
import os
import subprocess
import sys
import time
from glob import glob
from os import path as osp
from typing import Callable, List, Optional

from mim.utils import (get_installed_path, highlighted_error, is_installed,
                       module_full_name)
from mmcv import Config

from siatune.utils import ref_raw_args
from ..base import BaseTask
from ..builder import TASKS
from ._bystander import reporter_factory


@TASKS.register_module()
class BystanderTrainBasedTask(BaseTask):

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
        self._raw_args: List[str] = []
        self._rewriters: List[dict] = rewriters

    def parse_args(self, *args, **kwargs) -> argparse.Namespace:
        """Parse and set the argss.

        Args:
            args (Sequence[str]): The args.
        """
        return argparse.Namespace()

    def _get_train_script(self, pkg_name: str) -> str:
        pkg_full_name = module_full_name(pkg_name)
        if pkg_full_name == '':
            msg = "Can't determine a unique "
            f'package given abbreviation {pkg_name}'
            raise ValueError(highlighted_error(msg))
        if not is_installed(pkg_full_name):
            msg = (f'The codebase {pkg_name} is not installed')
            raise RuntimeError(highlighted_error(msg))
        pkg_root = get_installed_path(pkg_full_name)
        # tools will be put in package/.mim in PR #68
        train_script = osp.join(pkg_root, '.mim', 'tools', 'train.py')
        if not osp.exists(train_script):
            _alternative = osp.join(pkg_root, 'tools', 'train.py')
            if osp.exists(_alternative):
                train_script = _alternative
            else:
                raise RuntimeError("Can't find train script")
        return train_script

    def _get_work_dir(self, raw_args: List[str], config_idx: int = 0) -> str:
        work_dir, _ = ref_raw_args(raw_args, '--work-dir')
        if work_dir:
            return work_dir.pop()
        # TODO
        cfg = Config.fromfile(raw_args[config_idx])
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
            files = glob(osp.join(work_dir, '*' + log_ext))
            if files:
                break
            elif time.time() > t_bound:
                raise RuntimeError
        if len(files) > 1:
            raise ValueError
        return files.pop()

    # TODO: attach ckpt linking bystander
    def _attach_bystander(self, raw_args: List[str],
                          metric: str) -> List[multiprocessing.Process]:
        bystanders: List[multiprocessing.Process] = []
        work_dir = self._get_work_dir(raw_args)
        log_file = self._get_deffered_log(work_dir)
        reporter = reporter_factory(log_file, metric)
        reporter.start()
        bystanders.append(reporter)
        return bystanders

    def run(self, *, raw_args: List[str], **kwargs) -> None:
        cmd: List[str]
        port: int = kwargs.get('port', 29500)

        launcher, _ = ref_raw_args(raw_args, '--launcher')
        assert len(launcher) < 2
        launcher = launcher.pop() if launcher else 'none'

        if launcher == 'none':
            cmd = [sys.executable, self._train_script]
        elif launcher == 'pytorch':

            cmd = [
                sys.executable,
                '-m',
                'torch.distributed.launch',
                f'--nproc_per_node={self.num_workers}',
                f'--master_port={port}',
                self._train_script,
            ]
        else:
            NotImplementedError
        cmd.extend(raw_args)

        worker = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            env=dict(os.environ, MASTER_PORT=str(port)))

        bystanders = self._attach_bystander(raw_args, self._metric)

        _, err = worker.communicate()
        if worker.returncode != 0:
            raise Exception(err)
        for bystander in bystanders:
            bystander.shutdown()
        return

    def context_aware_run(self,
                          searched_cfg,
                          backend='nccl',
                          **kwargs) -> None:
        """Gather and refine the information received by users and Ray.tune to
        execute the objective task.

        Args:
            searched_cfg (Config): The searched configs.
            backend (str):
                The backend for dist training. Defaults to 'nccl'.
            kwargs (**kwargs): The kwargs.
        """
        # set non blocking mode on the nccl backend
        # https://github.com/pytorch/pytorch/issues/50820
        if backend == 'nccl' and os.getenv('NCCL_BLOCKING_WAIT') is None:
            os.environ['NCCL_BLOCKING_WAIT'] = '0'
        return super().context_aware_run(
            searched_cfg,
            raw_args=self._raw_args,
            tune_launch_path=self.tune_launch_path,
            **kwargs)

    def create_trainable(self) -> Callable:
        """Get ray trainable task.

        Returns:
            Callable: The Ray trainable task.
        """

        return self.context_aware_run
