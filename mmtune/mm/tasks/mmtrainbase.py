import os
from abc import abstractmethod
from functools import partial

import mmcv
import ray
import torch
from ray.tune.integration.tensorflow import DistributedTrainableCreator

from .base import BaseTask
from .builder import TASKS


@TASKS.register_module()
class MMTrainBasedTask(BaseTask):

    @classmethod
    @abstractmethod
    def build_model(cls, cfg: mmcv.Config, **kwargs) -> torch.nn.Module:
        pass

    @classmethod
    @abstractmethod
    def build_dataset(cls, cfg: mmcv.Config,
                      **kwargs) -> torch.utils.data.Dataset:
        pass

    @classmethod
    @abstractmethod
    def train_model(cls, model: torch.nn.Module,
                    dataset: torch.utils.data.Dataset, cfg: mmcv.Config,
                    **kwargs) -> None:
        pass

    @classmethod
    def create_trainable(cls, backend: str = 'nccl') -> ray.tune.trainable:
        assert backend in ['gloo', 'nccl']
        if backend == 'nccl' and os.getenv('NCCL_BLOCKING_WAIT') is None:
            os.environ['NCCL_BLOCKING_WAIT'] = '0'

        return DistributedTrainableCreator(
            partial(
                cls.contextaware_run,
                status=dict(
                    base_cfg=cls.BASE_CFG,
                    args=cls.ARGS,
                    rewriters=cls.REWRITERS)),
            backend=backend,
            num_workers=cls.ARGS.num_workers,
            num_gpus_per_worker=cls.ARGS.num_cpus_per_worker,
            num_cpus_per_worker=cls.ARGS.num_cpus_per_worker)
