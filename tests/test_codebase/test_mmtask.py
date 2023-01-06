from unittest.mock import MagicMock, patch

import mmcls  # noqa: F401
import mmdet  # noqa: F401
import mmedit  # noqa: F401
import mmseg  # noqa: F401
import pytest

from siatune.codebase import (MIM, MMClassification, MMDetection, MMEditing,
                              MMSegmentation)
from siatune.codebase.mim import _EntrypointExecutor


@patch('mmcls.apis.train_model')
@patch('mmcls.datasets.build_dataset')
@patch('mmcls.models.build_classifier')
def test_mmcls(*mocks):
    task = MMClassification(args=['tests/data/config.py'], num_workers=1)
    task.run(args=task.args)


@patch('mmdet.apis.train_detector')
@patch('mmdet.datasets.build_dataset')
@patch('mmdet.models.build_detector')
def test_mmdet(*mocks):
    task = MMDetection(args=['tests/data/config.py'], num_workers=1)
    task.run(args=task.args)


@patch('mmedit.apis.train_model')
@patch('mmedit.datasets.build_dataset')
@patch('mmedit.models.build_model')
def test_mmedit(*mocks):
    task = MMEditing(args=['tests/data/config.py'], num_workers=1)
    task.run(args=task.args)


@patch('mmseg.apis.train_segmentor')
@patch('mmseg.datasets.build_dataset')
@patch('mmseg.models.build_segmentor')
def test_mmseg(*mocks):
    task = MMSegmentation(args=['tests/data/config.py'], num_workers=1)
    task.run(args=task.args)


@patch('siatune.utils.get_train_script')
def test_entrypoint(mock_get_train_script):
    mock_get_train_script.return_value = '../data/entrypoint.py'
    entrypoint_executor = _EntrypointExecutor('test', [])
    with pytest.raises(TypeError) as ex:
        entrypoint_executor.execute()
    assert ex.value.args[0] == 'Test'


@patch('siatune.codease.mm._EntrypointExecutor')
def test_mim(_MockEntrypointExecutor):

    def ex():
        raise Exception('Test')

    mock_executor = MagicMock()
    mock_executor.execute = ex
    _MockEntrypointExecutor.return_value = mock_executor
    task = MIM(
        pkg_name='.',
        args=[''],
    )
    with pytest.raises(TypeError) as ex:
        task.run(args=task.args)
    assert ex.value.args[0] == 'Test'
