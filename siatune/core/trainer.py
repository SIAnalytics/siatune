# Copyright (c) SI-Analytics. All rights reserved.
from typing import Callable

import ray
import torch
from ray.air.config import ScalingConfig
from ray.train._internal.utils import get_address_and_port

from siatune.utils import set_env_vars


class DataParallelTrainCreator:

    def __init__(self,
                 trainable: Callable,
                 num_cpus_per_worker: int = 1,
                 num_workers: int = 1):
        self.trainable = trainable
        num_rest_worker: int
        if num_workers > 1:
            num_rest_worker = num_workers - 1
        else:
            num_rest_worker = 0
        self._resources = ScalingConfig(
            trainer_resources=dict(
                CPU=num_cpus_per_worker, GPU=int(torch.cuda.is_available())),
            use_gpu=torch.cuda.is_available(),
            num_workers=num_rest_worker,
            resources_per_worker=dict(CPU=num_cpus_per_worker))
        return

    def train(self, *args, **kwargs):
        num_workers = self.resources.num_workers
        num_cpus_per_worker = self.resources.resources_per_worker.get('CPU')
        addr, port = get_address_and_port()

        def job(rank):
            set_env_vars(rank, num_workers, addr, port)
            self.trainable(*args, **kwargs)

        job(0)
        remote_job = ray.remote(job).options(
            num_cpus=num_cpus_per_worker, num_gpus=1)
        ray.get([remote_job.remote(rank) for rank in range(1, num_workers)])
        return

    @property
    def resources(self) -> ScalingConfig:
        return self._resources
