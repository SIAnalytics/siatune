import os
from abc import abstractmethod
from functools import partial

import mmcv
import ray
import torch
from ray.tune.integration.torch import DistributedTrainableCreator

from mmtune.mm.context import ContextManager
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
    def contextaware_run(cls, status, backend, *args, **kwargs) -> None:
        from mmtune.mm.tasks import hooks  # noqa F401
        if backend == 'nccl' and os.getenv('NCCL_BLOCKING_WAIT') is None:
            os.environ['NCCL_BLOCKING_WAIT'] = '0'
        context_manager = ContextManager(**status)
        return context_manager(cls.run)(*args, **kwargs)

    @classmethod
    def create_trainable(cls, backend: str = 'nccl') -> ray.tune.trainable:
        assert backend in ['gloo', 'nccl']

        return DistributedTrainableCreator(
            partial(
                cls.contextaware_run,
                dict(
                    base_cfg=cls.BASE_CFG,
                    args=cls.ARGS,
                    rewriters=cls.REWRITERS), backend),
            backend=backend,
            num_workers=cls.ARGS.num_workers,
            num_gpus_per_worker=cls.ARGS.num_gpus_per_worker,
            num_cpus_per_worker=cls.ARGS.num_cpus_per_worker)
