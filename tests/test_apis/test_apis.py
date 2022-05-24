import os
import tempfile
from unittest.mock import MagicMock

import mmcv
from ray.tune.trainable import Trainable

from mmtune.apis import log_analysis, tune


def test_log_analysis():
    mock_analysis = MagicMock()

    task_config = mmcv.Config(dict(model=dict(type='TempModel')))

    tune_config = mmcv.Config(
        dict(
            scheduler=dict(
                type='AsyncHyperBandScheduler',
                time_attr='training_iteration',
                max_t=20,
                grace_period=2),
            metric='accuracy',
            mode='max',
        ))

    mock_analysis.best_config = task_config
    mock_analysis.best_result = dict(accuracy=50)
    mock_analysis.best_logdir = 'temp_log_dir'
    mock_analysis.results = [dict(accuracy=50)]

    with tempfile.TemporaryDirectory() as tmpdir:
        log_analysis(mock_analysis, tune_config, task_config, tmpdir)
        assert os.path.exists(os.path.join(tmpdir, 'tune_config.py'))
        assert os.path.exists(os.path.join(tmpdir, 'task_config.py'))


def test_tune():

    class TestTrainable(Trainable):

        def step(self):
            result = {'name': self.trial_name, 'trial_id': self.trial_id}
            return result

    tune_config = mmcv.Config(
        dict(
            scheduler=dict(
                type='AsyncHyperBandScheduler',
                time_attr='training_iteration',
                max_t=3,
                grace_period=1),
            metric='accuracy',
            mode='max',
            num_samples=1))

    mock_task_processor = MagicMock()
    mock_task_processor.create_trainable.return_value = TestTrainable
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_task_processor.args.work_dir = tmpdir
        mock_task_processor.args.num_workers = 1
        mock_task_processor.args.num_cpus_per_worker = 1
        mock_task_processor.args.num_gpus_per_worker = 0
        tune(mock_task_processor, tune_config, 'exp_name')
