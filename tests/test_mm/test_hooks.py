import os
from typing import Dict
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from ray import tune

from mmtune.mm.hooks import RayCheckpointHook, RayTuneLoggerHook


def test_raycheckpointhook():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('gloo', rank=0, world_size=1)

    hook = RayCheckpointHook(
        interval=1,
        by_epoch=True,
    )
    mock_runner = MagicMock()
    mock_runner.inner_iter = 3
    mock_runner.iter = 5
    mock_runner.epoch = 5

    mock_runner.model = torch.nn.Linear(2, 2)
    mock_runner.optimizer = torch.optim.Adam(mock_runner.model.parameters())

    hook._save_checkpoint(mock_runner)
    assert os.path.exists('ray_checkpoint.pth')


@patch.object(RayTuneLoggerHook, 'get_loggable_tags')
def test_raytuneloggerhook(mock_get_loggable_tags):

    def trainable(config: Dict):
        mock_get_loggable_tags.return_value = config
        mock_runner = MagicMock()
        loggerhook = RayTuneLoggerHook(filtering_key='test')

        for itr in range(16):
            mock_runner.iter = itr
            loggerhook.log(mock_runner)

    tune.run(trainable, config={'test': tune.uniform(-5, -1)}, num_samples=10)
