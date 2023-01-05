# Copyright (c) SI-Analytics. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Callable, Sequence

from siatune.core import DistTorchLauncher
from .base import BaseTask
from .builder import TASKS


@TASKS.register_module()
class MMBaseTask(BaseTask, metaclass=ABCMeta):
    """Wrap the apis of open mm train-based projects."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.num_gpus_per_worker == 1
        self.launcher = DistTorchLauncher(
            self.num_cpus_per_worker,
            self.num_workers,
        )

    def run(self, *args, **kwargs):
        self.launcher.launch(self.execute, *args, **kwargs)

    @abstractmethod
    def execute(self, args: Sequence[str]):
        pass

    def create_trainable(self) -> Callable:
        """Get a :class:`DataParallelTrainer` instance.

        Returns:
            Callable: Callable object to optimize hyperparameter.
        """
        return self.launcher.reserve(self.context_aware_run)
