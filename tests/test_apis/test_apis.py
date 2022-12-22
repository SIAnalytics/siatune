import os
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import mmcv

from siatune.apis import log_analysis


def test_log_analysis():
    mock_results = MagicMock()

    with TemporaryDirectory() as tmpdir:
        with TemporaryDirectory() as logdir:
            mock_results.get_best_result.return_value = MagicMock(
                log_dir=logdir,
                config=mmcv.Config(dict(model=dict(type='TestModel'))))
            log_analysis(mock_results, log_dir=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, 'best_trial'))
