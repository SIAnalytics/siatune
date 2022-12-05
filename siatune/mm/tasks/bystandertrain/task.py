from ..builder import TASKS

from ..blackbox import BlackBoxTask
import os
from os import path as osp
import argparse
import subprocess
from utils import revert_args, module_full_name, get_installed_path
from typing import Optional, List



@TASKS.register_module()
class BystanderTrainBasedTask(BlackBoxTask):
    def __init__(self, pkg_name:str, rewriters: List[dict] = []) -> None:
        """Initialize the task.

        Args:
            pkg_name (str): 
            rewriters (List[dict]):
                Context redefinition pipeline. Defaults to [].
        """
        self.train_script = self._get_train_script(pkg_name)
        self._args: Optional[argparse.Namespace] = None
        self._rewriters: List[dict] = rewriters

    def _get_train_script(self, pkg_name: str) -> str:
        pkg_full_name = module_full_name(pkg_name)
        if pkg_full_name == '':
            raise ValueError(f"Can't determine a unique package given abbreviation {pkg_name}")
        pkg_root = get_installed_path(pkg_full_name)
        train_script = osp.join(pkg_root, '.mim', 'tools', 'train.py')
        if not osp.exists(train_script):
            raise ValueError("Can't find train script")
        return train_script




    def run(self, *, args: argparse.Namespace, **kwargs) -> None:



        if args.launcher == 'none':
            cmd = ['python', self.train_script]
        else args.launcher == 'pytorch': 
            cmd = [
                'python', '-m', 'torch.distributed.launch',
                f'--nproc_per_node={self.num_workers}', f'--master_port={port}',
                train_script, config_path
            ]
        else:
            NotImplementedError
            

        args = revert_args(args)
        worker = subprocess.Popen(cmd1)
