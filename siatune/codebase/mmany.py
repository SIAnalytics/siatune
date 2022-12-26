# Copyright (c) SI-Analytics. All rights reserved.
from .blackbox import BaseTask
from mim.utils import (get_installed_path, highlighted_error, is_installed,
                        module_full_name)
from os import path as osp
from typing import Sequence, Callable
import argparse
from siatune.utils import ref_raw_args
from importlib.machinery import SourceFileLoader
from types import ModuleType
import ray
from ray import tune
from .builder import TASKS


@TASKS.register_module()
class MMAny(BaseTask):
    def __init__(self, pkg_name:str, **kwargs):
        self._entrypoint: ModuleType = self._get_entrypoint(pkg_name)
        super().__init__(self, **kwargs)

    def _get_entrypoint(self, pkg_name: str) -> ModuleType:
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
        return SourceFileLoader('train', train_script).load_module()


    def parse_args(self, args: Sequence[str]) -> argparse.Namespace:
        return argparse.Namespace()

    def run(self, *, raw_args:Sequence[str]) -> None:
        import sys as _sys
        _sys.argv[1:] = raw_args

        launcher, _ = ref_raw_args(raw_args, '--launcher')
        assert len(launcher) < 2
        launcher = launcher.pop() if launcher else 'none'

        if launcher == 'none':
            self._entrypoint.main()
        else:
            assert self.num_gpus_per_worker == 1
            self._dist_run(self._entrypoint, self.num_workers)

    def _dist_run(self, entrypoint: ModuleType, world_size:int, addr: str="127.0.0.1", port: int=29500) -> None:

        def job(rank: int):
            import os as _os
            _os.environ['MASTER_ADDR'] = addr
            _os.environ['MASTER_PORT'] = str(port)
            _os.environ['RANK'] = str(rank)
            _os.environ['LOCAL_RANK'] = str(rank)
            _os.environ['WORLD_SIZE'] = str(world_size)

            entrypoint.main()
            return
        
        remote_job = ray.remote(job, num_cpus=self.num_cpus_per_worker,
            num_gpus=self.num_gpus_per_worker)

        ray.get([remote_job.remote(rank) for rank in range(world_size)])
        return
    
    def create_trainable(self) -> Callable:
         """Get ray trainable task.

         Returns:
             Callable: The Ray trainable task.
         """

         return tune.with_resources(
             self.context_aware_run,
             dict(
                 CPU=self.num_workers * self.num_cpus_per_worker,
                 GPU=self.num_workers * self.num_gpus_per_worker))
