# Copyright (c) SI-Analytics. All rights reserved.
from typing import Callable

import ray
import ray.tune.with_resources as reserve_resources
from ray.air.config import ScalingConfig

from siatune.utils import set_env_vars


class DistTrainer:

    def __init__(self,
                 trainable: Callable,
                 num_cpus_per_worker: int = 1,
                 num_workers: int = 1):
        self.trainable = trainable
        self.num_cpus_per_worker = num_cpus_per_worker
        self.num_workers = num_workers

    def train(self):

        def job(rank):
            set_env_vars(rank, self.num_workers)
            self.trainable()

        remote_job = ray.remote(job).options(
            num_cpus=self.num_cpus_per_worker, num_gpus=self.num_workers)
        ray.get([remote_job.remote(rank) for rank in range(self.num_workers)])
        return


def create_dist_trainer(
    trainable: Callable,
    num_cpus_per_worker: int = 1,
    num_workers: int = 1,
) -> Callable:
    resources = ScalingConfig(
        trainer_resources=dict(),
        use_gpu=True,
        num_workers=num_workers,
        resources_per_worker=dict(CPU=num_cpus_per_worker))

    return reserve_resources(
        DistTrainer(trainable, num_cpus_per_worker, num_workers).train,
        resources)
