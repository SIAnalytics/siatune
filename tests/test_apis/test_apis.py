import os
import tempfile
from unittest.mock import MagicMock

import mmcv

from siatune.apis import log_analysis


def test_log_analysis():
    mock_analysis = MagicMock()

    task_config = mmcv.Config(dict(model=dict(type='TempModel')))

    mock_analysis.best_config = task_config
    mock_analysis.best_result = dict(accuracy=50)
    mock_analysis.best_logdir = 'temp_log_dir'
    mock_analysis.results = [dict(accuracy=50)]

    with tempfile.TemporaryDirectory() as tmpdir:
        tune_config = mmcv.Config(
            dict(
                scheduler=dict(
                    type='AsyncHyperBandScheduler',
                    time_attr='training_iteration',
                    max_t=20,
                    grace_period=2),
                metric='accuracy',
                mode='max',
                work_dir=tmpdir))

        log_analysis(mock_analysis, tune_config, task_config, tmpdir)
        assert os.path.exists(os.path.join(tmpdir, 'tune_config.py'))
        assert os.path.exists(os.path.join(tmpdir, 'task_config.py'))
