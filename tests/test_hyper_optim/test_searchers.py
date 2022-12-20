import pytest
from ray import tune

from siatune.tune.searchers import SEARCHERS, build_searcher


def test_build_searcher():

    @SEARCHERS.register_module()
    class TestSearcher:
        pass

    assert isinstance(build_searcher({'type': 'TestSearcher'}), TestSearcher)


@pytest.fixture
def config():
    return dict(
        steps=10, width=tune.uniform(0, 20), height=tune.uniform(-100, 100))


@pytest.fixture
def trainable():

    def _trainable(config):
        width, height = config['width'], config['height']
        for step in range(config['steps']):
            intermediate_score = (0.1 +
                                  width * step / 100)**(-1) + height * 0.1
            tune.report(iterations=step, mean_loss=intermediate_score)

    return _trainable


def test_blend(trainable, config):
    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        search_alg=build_searcher(dict(type='BlendSearch')),
        num_samples=2,
        config=config)


def test_bohb(trainable, config):
    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        search_alg=build_searcher(dict(type='TuneBOHB')),
        num_samples=2,
        config=config)


def test_cfo(trainable, config):
    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        search_alg=build_searcher(dict(type='CFO')),
        num_samples=2,
        config=config)


def test_hyperopt(trainable, config):
    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        search_alg=build_searcher(dict(type='HyperOptSearch')),
        num_samples=2,
        config=config)


def test_nevergrad(trainable, config):
    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        search_alg=build_searcher(dict(type='NevergradSearch', budget=1)),
        num_samples=2,
        config=config)

    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        search_alg=build_searcher(
            dict(type='NevergradSearch', optimizer='PSO', budget=1)),
        num_samples=2,
        config=config)


def test_optuna(trainable, config):
    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        search_alg=build_searcher(dict(type='OptunaSearch')),
        num_samples=2,
        config=config)
