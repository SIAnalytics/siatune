# Copyright (c) SI-Analytics. All rights reserved.
import os


def set_env_vars(rank: int,
                 world_size: int,
                 addr: str = '127.0.0.1',
                 port: int = 29500):
    """Sets environment variables for distributed training.

    Args:
        rank (int):
            The rank of the current process.
        world_size (int):
            The total number of processes
            in the distributed training.
        addr (str, optional):
            The address of the master process.
            Defaults to '127.0.0.1'.
        port (int, optional):
            The port number used by the master process.
            Defaults to 29500.
    """
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = str(port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
