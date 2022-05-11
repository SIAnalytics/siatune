import os
from abc import abstractmethod

import mmcv
import ray
import torch
from ray.tune.integration.tensorflow import DistributedTrainableCreator

from .base import BaseTask
from .builder import TASKS


@TASKS.register_module()
class MMTrainBasedTask(BaseTask):

    @staticmethod
    @abstractmethod
    def build_model(cfg: mmcv.Config, **kwargs) -> torch.nn.Module:
        pass

    @staticmethod
    @abstractmethod
    def build_dataset(cfg: mmcv.Config, **kwargs) -> torch.utils.data.Dataset:
        pass

    @staticmethod
    @abstractmethod
    def train_model(model: torch.nn.Module, dataset: torch.utils.data.Dataset,
                    cfg: mmcv.Config, **kwargs) -> None:
        pass

    @staticmethod
    def create_trainable(backend: str = 'nccl') -> ray.tune.trainable:
        assert backend in ['gloo', 'nccl']
        if backend == 'nccl' and os.getenv('NCCL_BLOCKING_WAIT') is None:
            os.environ['NCCL_BLOCKING_WAIT'] = '0'

        return DistributedTrainableCreator(
            MMTrainBasedTask.run,
            backend=backend,
            num_workers=MMTrainBasedTask.ARGS.num_workers,
            num_gpus_per_worker=MMTrainBasedTask.ARGS.num_cpus_per_worker,
            num_cpus_per_worker=MMTrainBasedTask.ARGS.num_cpus_per_worker)
