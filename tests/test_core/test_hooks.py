from typing import Dict
from unittest.mock import MagicMock, patch

from ray import tune

from siatune.core.hooks import RayTuneReporterHook


@patch.object(RayTuneReporterHook, 'get_loggable_tags')
def test_RayTuneReporterHook(mock_get_loggable_tags):

    def trainable(config: Dict):
        mock_get_loggable_tags.return_value = config
        mock_runner = MagicMock()
        loggerhook = RayTuneReporterHook(filtering_key='test')

        for itr in range(16):
            mock_runner.iter = itr
            loggerhook.log(mock_runner)

    tune.run(trainable, config={'test': tune.uniform(-5, -1)}, num_samples=10)
