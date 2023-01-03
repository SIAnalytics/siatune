# Copyright (c) SI-Analytics. All rights reserved.
from typing import Sequence

import torch
from ray.air.config import ScalingConfig
from ray.train.data_parallel_trainer import DataParallelTrainer

from siatune.core import DataParallelTrainCreator
from siatune.tune import MMBackendConfig
from ..base import BaseTask
from ..builder import TASKS
from ._entrypoint import EntrypointRunner


@TASKS.register_module()
class MIM(BaseTask):

    def __init__(self, pkg_name: str, **kwargs):
        self._pkg_name = pkg_name
        super().__init__(should_parse=False, **kwargs)
        assert self.num_gpus_per_worker == 1
        self.dist_creator = DataParallelTrainCreator(
            self._run,
            num_cpus_per_worker=self.num_cpus_per_worker,
            num_workers=self.num_workers)

    def parse_args(self, *args, **kwargs) -> None:
        return None

    def run(self, *arg, **kwargs):
        self.dist_creator.train(*arg, **kwargs)

    def _run(self, *, args: Sequence[str], **kwargs) -> None:
        runner = EntrypointRunner(self._pkg_name, args)
        runner.run()

    def create_trainable(self) -> DataParallelTrainer:
        """Get a :class:`DataParallelTrainer` instance.

        Returns:
            DataParallelTrainer: Trainer to optimize hyperparameter.
        """

        return DataParallelTrainer(
            self.context_aware_run,
            backend_config=MMBackendConfig(),
            scaling_config=ScalingConfig(
                trainer_resources=dict(CPU=self.num_cpus_per_worker),
                num_workers=self.num_workers,
                use_gpu=torch.cuda.is_available()))
