# Copyright (c) SI-Analytics. All rights reserved.
import os
from abc import ABCMeta, abstractmethod
from functools import partial

import mmcv
import torch
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer

from .base import BaseTask
from .builder import TASKS


@TASKS.register_module()
class MMTrainBasedTask(BaseTask, metaclass=ABCMeta):
    """Wrap the apis of open mm train-based projects."""

    @abstractmethod
    def build_model(self, cfg: mmcv.Config, **kwargs) -> torch.nn.Module:
        """Build the model from configs.

        Args:
            cfg (Config): The configs.
        Returns:
            torch.nn.Module: The model.
        """
        pass

    @abstractmethod
    def build_dataset(self, cfg: mmcv.Config,
                      **kwargs) -> torch.utils.data.Dataset:
        """Build the dataset from configs.

        Args:
            cfg (Config): The configs.
        Returns:
            torch.utils.data.Dataset: The dataset.
        """
        pass

    @abstractmethod
    def train_model(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        cfg: mmcv.Config,
        **kwargs,
    ) -> None:
        """Train the model.

        Args:
            model (torch.nn.Module): The model.
            dataset (torch.utils.data.Dataset): The dataset.
            cfg (Config): The configs.
        """
        pass

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
        return super().context_aware_run(searched_cfg, **kwargs)

    def create_trainable(
        self,
        backend: str = 'nccl',
    ) -> TorchTrainer:
        """Get ray trainable task.

        Args:
            backend (str): The backend for distributed training.
                Defaults to 'nccl'.

        Returns:
            TorchTrainer: The trainable task.
        """

        assert backend in ['gloo', 'nccl']

        return TorchTrainer(
            partial(self.context_aware_run, backend=backend),
            scaling_config=ScalingConfig(
                num_workers=2,
                use_gpu=True,
                resources_per_worker=dict(
                    CPU=self.num_cpus_per_worker,
                    GPU=self.num_gpus_per_worker)))
