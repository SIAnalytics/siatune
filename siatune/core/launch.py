# Copyright (c) SI-Analytics. All rights reserved.
from typing import Callable

import ray
import torch
from ray import tune
from ray.train._internal.utils import get_address_and_port
from ray.tune import with_resources as reserve_resources

from siatune.utils import set_env_vars


class DistTorchLauncher:

    def __init__(self, num_cpus_per_worker: int = 1, num_workers: int = 1):
        self.num_workers = num_workers
        self.num_cpus_per_worker = num_cpus_per_worker
        self.num_gpus_per_worker = int(torch.cuda.is_available())
        self.resources = tune.PlacementGroupFactory([
            dict(CPU=self.num_cpus_per_worker, GPU=self.num_gpus_per_worker),
        ] * num_workers)

    def launch(self, func: Callable, *args, **kwargs):
        addr, port = get_address_and_port()

        def job(rank):
            set_env_vars(rank, self.num_workers, addr, port)
            func(*args, **kwargs)

        remote_job = ray.remote(job).options(
            num_cpus=self.num_cpus_per_worker,
            num_gpus=self.num_gpus_per_worker)
        futures = [
            remote_job.remote(rank) for rank in range(1, self.num_workers)
        ]
        job(rank=0)
        ray.get(futures)

    def reserve(self, trainable: Callable):
        return reserve_resources(trainable, self.resources)
