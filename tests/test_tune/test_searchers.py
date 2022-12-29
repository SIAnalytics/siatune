from tempfile import TemporaryDirectory

import pytest
from mmengine import Config
from ray.air import session

from siatune.tune import Tuner
from siatune.tune.searchers import SEARCHERS, build_searcher


def test_build_searcher():

    @SEARCHERS.register_module()
    class TestSearcher:
        pass

    cfg = dict(type='TestSearcher')
    assert isinstance(build_searcher(cfg), TestSearcher)


@pytest.fixture
def trainable():

    def _trainable(config):
        config = config['train_loop_config']
        for step in range(config['iter']):
            loss = (0.1 + config['x'] * step / 100)**-1 + config['y'] * 0.1
            session.report(dict(loss=loss))

    return _trainable


@pytest.fixture
def param_space():
    return dict(
        iter=10,
        x=dict(type='Uniform', lower=0, upper=10),
        y=dict(type='Uniform', lower=0, upper=10))


# TODO: Fix BlendSearch
@pytest.mark.parametrize('type',
                         ['CFO', 'HyperOptSearch', 'OptunaSearch', 'TuneBOHB'])
def test_searcher(trainable, param_space, type):
    with TemporaryDirectory() as tmpdir:
        Tuner(
            trainable,
            tmpdir,
            param_space=param_space,
            tune_cfg=dict(metric='loss', mode='min', num_samples=2),
            searcher=dict(type=type),
            cfg=Config()).tune()


def test_nevergrad(trainable, param_space):
    with TemporaryDirectory() as tmpdir:
        Tuner(
            trainable,
            tmpdir,
            param_space=param_space,
            tune_cfg=dict(metric='loss', mode='min', num_samples=2),
            searcher=dict(type='NevergradSearch', budget=1),
            cfg=Config()).tune()

    with TemporaryDirectory() as tmpdir:
        Tuner(
            trainable,
            tmpdir,
            param_space=param_space,
            tune_cfg=dict(metric='loss', mode='min', num_samples=2),
            searcher=dict(type='NevergradSearch', optimizer='PSO', budget=1),
            cfg=Config()).tune()
