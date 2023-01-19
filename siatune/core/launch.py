# Copyright (c) SI-Analytics. All rights reserved.
from typing import Callable

import ray
import torch
from ray import tune
from ray.train._internal.utils import get_address_and_port

from siatune.utils import set_env_vars


class DistributedTorchLauncher:
    """Distributed Training Launcher for PyTorch.

    This class provides a simple
    interface for launching distributed
    training jobs with PyTorch.
    It uses the Ray framework to manage
    the distributed environment
    and to parallelize the training process.

    Args:
        num_cpus_per_worker (int, optional):
            Number of CPU cores to allocate per worker.
            Defaults to 1.
        num_workers (int, optional):
            Number of workers to launch.
            Defaults to 1.

    Attributes:
        num_workers (int):
            Number of workers launched.
        num_cpus_per_worker (int):
            Number of CPU cores allocated per worker.
        num_gpus_per_worker (int):
            Number of GPUs available per worker.
        resources (tune.PlacementGroupFactory):
            Ray placement group for resource allocation.
    """

    def __init__(self, num_cpus_per_worker: int = 1, num_workers: int = 1):
        self.num_workers = num_workers
        self.num_cpus_per_worker = num_cpus_per_worker
        self.num_gpus_per_worker = int(torch.cuda.is_available())
        self.resources = tune.PlacementGroupFactory([
            dict(CPU=self.num_cpus_per_worker, GPU=self.num_gpus_per_worker),
        ] * num_workers)

    def launch(self, func: Callable, *args, **kwargs):
        """Launch the distributed training job.

        Args:
            func (Callable):
                Training function to be executed
                in the distributed environment.
            *args:
                Positional arguments to be passed
                to the training function.
            **kwargs:
                Keyword arguments to be passed
                to the training function.
        """
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
        # To facilitate session sharing,
        # the master job will be initiated by the trainer.
        ray.get(futures)
