import os
from abc import abstractmethod
from functools import partial

import mmcv
import ray
import torch
from ray.tune.integration.torch import DistributedTrainableCreator

from .base import BaseTask
from .builder import TASKS


@TASKS.register_module()
class MMTrainBasedTask(BaseTask):

    @abstractmethod
    def build_model(self, cfg: mmcv.Config, **kwargs) -> torch.nn.Module:
        pass

    @abstractmethod
    def build_dataset(self, cfg: mmcv.Config,
                      **kwargs) -> torch.utils.data.Dataset:
        pass

    @abstractmethod
    def train_model(self, model: torch.nn.Module,
                    dataset: torch.utils.data.Dataset, cfg: mmcv.Config,
                    **kwargs) -> None:
        pass

    def context_aware_run(self,
                          *searched_cfg,
                          backend='nccl',
                          **context) -> None:
        if backend == 'nccl' and os.getenv('NCCL_BLOCKING_WAIT') is None:
            os.environ['NCCL_BLOCKING_WAIT'] = '0'
        return super().context_aware_run(*searched_cfg, **context)

    def create_trainable(self,
                         backend: str = 'nccl',
                         num_workers: int = 1,
                         num_gpus_per_worker: int = 1,
                         num_cpus_per_worker: int = 1) -> ray.tune.trainable:
        assert backend in ['gloo', 'nccl']

        return DistributedTrainableCreator(
            partial(
                self.context_aware_run,
                backend=backend,
            ),
            backend=backend,
            num_workers=num_workers,
            num_gpus_per_worker=num_gpus_per_worker,
            num_cpus_per_worker=num_cpus_per_worker)
