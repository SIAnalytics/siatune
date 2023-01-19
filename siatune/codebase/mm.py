# Copyright (c) SI-Analytics. All rights reserved.
from abc import ABCMeta
from copy import deepcopy
from typing import Callable

from ray.tune import with_resources

from siatune.core import ContextManager, DistributedTorchLauncher
from siatune.utils import ImmutableContainer
from .base import BaseTask
from .builder import TASKS


@TASKS.register_module()
class MMBaseTask(BaseTask, metaclass=ABCMeta):
    """Wrap the apis of open mm train-based projects."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.num_gpus_per_worker == 1
        self.launcher = DistributedTorchLauncher(
            self.num_cpus_per_worker,
            self.num_workers,
        )

    def create_trainable(self) -> Callable:
        """Get a trainable task.

        Returns:
            Callable: Callable object to optimize hyperparameter.
        """
        return with_resources(self.context_aware_run, self.launcher.resources)

    def context_aware_run(self, searched_cfg: dict):
        """Gather and refine the information received by users and Ray.tune to
        execute the objective task.

        Args:
            searched_cfg (Dict): The searched configuration.
        """

        context_manager = ContextManager(self.rewriters)
        context = dict(
            args=deepcopy(self.args),
            searched_cfg=deepcopy(ImmutableContainer.decouple(searched_cfg)),
        )
        return context_manager(self.dist_run)(**context)

    def dist_run(self, *args, **kwargs):
        self.launcher.launch(self.run, *args, **kwargs)
