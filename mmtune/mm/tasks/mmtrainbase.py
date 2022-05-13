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

    def contextaware_run(self, status, backend, *args, **kwargs) -> None:
        from mmtune.mm import hooks  # noqa F401
        if backend == 'nccl' and os.getenv('NCCL_BLOCKING_WAIT') is None:
            os.environ['NCCL_BLOCKING_WAIT'] = '0'
        context_manager = ContextManager(**status)
        return context_manager(self.run)(*args, **kwargs)

    def create_trainable(self, backend: str = 'nccl') -> ray.tune.trainable:
        assert backend in ['gloo', 'nccl']

        return DistributedTrainableCreator(
            partial(
                self.contextaware_run,
                dict(
                    base_cfg=self.base_cfg,
                    args=self.args,
                    rewriters=self.rewriters), backend),
            backend=backend,
            num_workers=self.args.num_workers,
            num_gpus_per_worker=self.args.num_cpus_per_worker,
            num_cpus_per_worker=self.args.num_cpus_per_worker)
