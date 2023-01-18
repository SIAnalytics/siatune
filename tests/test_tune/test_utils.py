import inspect
from unittest.mock import MagicMock

import pytest

from siatune.tune.utils import NAME_CREATOR


@pytest.fixture
def trial():
    return MagicMock(
        trainable_name='test',
        trial_id='trial_id',
        experiment_tag='experiment_tag')


def test_trial_id(trial):
    tmpl = NAME_CREATOR.get('trial_id')
    assert inspect.isfunction(tmpl)
    assert tmpl.__name__ == 'trial_id'
    assert tmpl(trial) == trial.trial_id


def test_experiment_tag(trial):
    tmpl = NAME_CREATOR.get('experiment_tag')
    assert inspect.isfunction(tmpl)
    assert tmpl.__name__ == 'experiment_tag'
    assert tmpl(trial) == trial.experiment_tag
