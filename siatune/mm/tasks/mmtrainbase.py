# Copyright (c) SI-Analytics. All rights reserved.
from abc import ABCMeta, abstractmethod

import mmcv
import torch
from ray.air.config import ScalingConfig
from ray.train.data_parallel_trainer import DataParallelTrainer

from siatune.ray.config import CustomBackendConfig
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

    def create_trainable(self) -> DataParallelTrainer:
        """Get ray trainable task.

        Args:
            backend (str): The backend for distributed training.
                Defaults to 'nccl'.

        Returns:
            TorchTrainer: The trainable task.
        """
        assert self.num_workers == self.num_gpus_per_worker, (
            '`num_workers` must be equal to `num_gpus_per_worker`.')

        return DataParallelTrainer(
            self.context_aware_run,
            backend_config=CustomBackendConfig(),
            scaling_config=ScalingConfig(
                trainer_resources=dict(
                    CPU=self.num_cpus_per_worker,
                    GPU=self.num_gpus_per_worker),
                num_workers=self.num_workers,
                use_gpu=torch.cuda.is_available()))
