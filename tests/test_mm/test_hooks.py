import os
from unittest.mock import MagicMock

import torch
import torch.distributed as dist

from mmtune.mm.hooks import RayCheckpointHook


def test_hook():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('gloo', rank=0, world_size=1)

    hook = RayCheckpointHook(
        interval=1,
        by_epoch=True,
        out_dir='/tmp/ray_checkpoint',
        mode='min',
        metric_name='loss',
        max_concurrent=1,
        checkpoint_metric=True,
        checkpoint_at_end=True,
    )
    mock_runner = MagicMock()
    mock_runner.inner_iter = 3
    mock_runner.iter = 5

    cur_iter = hook.get_iter(mock_runner, False)
    assert cur_iter == 6
    cur_iter = hook.get_iter(mock_runner, True)
    assert cur_iter == 4

    mock_runner.model = torch.nn.Linear(2, 2)

    hook._save_checkpoint(mock_runner)
