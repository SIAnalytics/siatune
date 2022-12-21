# Copyright (c) SI-Analytics. All rights reserved.
# Modified from https://github.com/ray-project/ray/blob/ray-2.1.0/python/ray/train/torch/config.py  # noqa

import logging
import os
from dataclasses import dataclass

import ray
import torch.distributed as dist
from ray.train._internal.utils import get_address_and_port
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import BackendConfig
from ray.train.torch.config import _set_nccl_network_interface, _TorchBackend

logger = logging.getLogger(__name__)


@dataclass
class MMBackendConfig(BackendConfig):
    """Configuration for torch process group setup."""

    @property
    def backend_cls(self):
        return _CustomTorchBackend


class _CustomTorchBackend(_TorchBackend):

    def on_start(self, worker_group: WorkerGroup,
                 backend_config: BackendConfig):
        if not dist.is_available():
            raise RuntimeError('Distributed torch is not available.')

        worker_group.execute(_set_nccl_network_interface)

        master_addr, master_port = worker_group.execute_single(
            0, get_address_and_port)

        def set_env_vars(addr, port, rank, world_size):
            os.environ['MASTER_ADDR'] = addr
            os.environ['MASTER_PORT'] = str(port)
            os.environ['RANK'] = str(rank)
            os.environ['LOCAL_RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(world_size)

        setup_futures = []
        for i in range(len(worker_group)):
            setup_futures.append(
                worker_group.execute_single_async(
                    i,
                    set_env_vars,
                    addr=master_addr,
                    port=master_port,
                    rank=i,
                    world_size=len(worker_group),
                ))
        ray.get(setup_futures)
