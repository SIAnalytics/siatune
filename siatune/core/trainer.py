# Copyright (c) SI-Analytics. All rights reserved.
from typing import Callable

import ray
import torch
from ray.air.config import ScalingConfig
from ray.train._internal.utils import get_address_and_port
from ray.tune import with_resources as reserve_resources

from siatune.utils import set_env_vars


class DataParallelTrainerCreator:

    def __init__(self,
                 trainable: Callable,
                 num_cpus_per_worker: int = 1,
                 num_workers: int = 1):

        self.trainable = trainable
        self.resources = ScalingConfig(
            trainer_resources=dict(),
            use_gpu=torch.cuda.is_available(),
            num_workers=num_workers,
            resources_per_worker=dict(CPU=num_cpus_per_worker))
        return

    def _train(self, *args, **kwargs):
        num_workers = self.resources.num_workers
        num_cpus_per_worker = self.resources.resources_per_worker.get('CPU')
        addr, port = get_address_and_port()

        def job(rank):
            set_env_vars(rank, num_workers, addr, port)
            self.trainable(*args, **kwargs)

        remote_job = ray.remote(job).options(
            num_cpus=num_cpus_per_worker, num_gpus=1)
        ray.get([remote_job.remote(rank) for rank in range(num_workers)])

    def create(self) -> Callable:
        return reserve_resources(self._train, self.resources)
