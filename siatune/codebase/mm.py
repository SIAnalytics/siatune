# Copyright (c) SI-Analytics. All rights reserved.
from abc import ABCMeta

from ray.train.data_parallel_trainer import DataParallelTrainer

from siatune.core import DataParallelTrainCreator
from .base import BaseTask
from .builder import TASKS


@TASKS.register_module()
class MMBaseTask(BaseTask, metaclass=ABCMeta):
    """Wrap the apis of open mm train-based projects."""

    def create_trainable(self) -> DataParallelTrainer:
        """Get a :class:`DataParallelTrainer` instance.

        Returns:
            DataParallelTrainer: Trainer to optimize hyperparameter.
        """
        return DataParallelTrainCreator(
            self.context_aware_run,
            num_cpus_per_worker=self.num_cpus_per_worker,
            num_workers=self.num_workers).create()
