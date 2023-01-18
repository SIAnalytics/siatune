import inspect

import pytest
from ray.tune.experiment import Trial

from siatune.tune.utils import NAME_TMPL


@pytest.fixture
def trial():
    return Trial(
        trainable_name='test',
        trial_id='trial_id',
        experiment_tag='experiment_tag')


def test_trial_id(trial):
    tmpl = NAME_TMPL.get('trial_id')
    assert inspect.isfunction(tmpl)
    assert tmpl.__name__ == 'trial_id'
    assert tmpl(trial) == trial.trial_id


def test_experiment_tag(trial):
    tmpl = NAME_TMPL.get('trial_id')
    assert inspect.isfunction(tmpl)
    assert tmpl.__name__ == 'experiment_tag'
    assert tmpl(trial) == trial.experiment_tag
