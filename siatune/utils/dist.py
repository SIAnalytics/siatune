# Copyright (c) SI-Analytics. All rights reserved.
import os


def set_env_vars(rank: int,
                 world_size: int,
                 addr: str = '127.0.0.1',
                 port: int = 29500):
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = str(port)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
