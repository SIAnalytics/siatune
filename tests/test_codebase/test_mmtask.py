from unittest.mock import MagicMock, patch

import pytest

from siatune.codebase import MIM
from siatune.codebase.mim import _EntrypointExecutor


@patch('siatune.codebase.mim.get_train_script')
def test_entrypoint(mock_get_train_script):
    mock_get_train_script.return_value = 'tests/data/entrypoint.py'
    entrypoint_executor = _EntrypointExecutor('test', [])
    with pytest.raises(Exception) as ex:
        entrypoint_executor.execute()
    assert ex.value.args[0] == 'Test'


@patch('siatune.codebase.mim._EntrypointExecutor')
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
    with pytest.raises(Exception) as ex:
        task.run(args=task.args)
    assert ex.value.args[0] == 'Test'
