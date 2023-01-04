# Copyright (c) SI-Analytics. All rights reserved.
from typing import Callable

import ray
import torch
from ray.air.config import ScalingConfig
from ray.train._internal.utils import get_address_and_port
from ray.tune import with_resources as reserve_resources

from siatune.utils import set_env_vars


class DistTorchLauncher:

    def __init__(self,
                 num_cpus_per_worker: int = 1,
                 num_workers: int = 1):

        num_remote_worker: int
        if num_workers > 1:
            num_remote_worker = num_workers - 1
        else:
            num_remote_worker = num_workers
        use_gpu: bool = torch.cuda.is_available()
                
        self._resources = ScalingConfig(
            trainer_resources=dict(CPU=num_cpus_per_worker, GPU=int(use_gpu)),
            use_gpu=use_gpu,
            num_workers=num_remote_worker,
            resources_per_worker=dict(CPU=num_cpus_per_worker))
        return

    @property
    def resources(self):
        return self._resources

    def launch(self, func: Callable, *args, **kwargs):
        num_workers = self.resources.num_workers + 1
        num_cpus_per_worker = self.resources.resources_per_worker.get('CPU')
        addr, port = get_address_and_port()

        def job(rank):
            set_env_vars(rank, num_workers, addr, port)
            func(*args, **kwargs)

        remote_job = ray.remote(job).options(
            num_cpus=num_cpus_per_worker, num_gpus=1)
        futures = [remote_job.remote(rank) for rank in range(1, num_workers)]
        func(rank=0)
        ray.get(futures)
        return

    def reserve(self, trainable: Callable):
        return reserve_resources(
            trainable,
            self.resources)