import pytest
from ray import tune

from mmtune.ray.searchers import (SEARCHERS, AxSearch, BlendSearch, CFOSearch,
                                  HyperOptSearch, NevergradSearch,
                                  TrustRegionSearcher, build_searcher)


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


def test_ax(trainable, config):
    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        search_alg=AxSearch(),
        num_samples=2,
        config=config)


def test_blend(trainable, config):
    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        search_alg=BlendSearch(),
        num_samples=2,
        config=config)


def test_cfo(trainable, config):
    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        search_alg=CFOSearch(),
        num_samples=2,
        config=config)


def test_hyperopt(trainable, config):
    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        search_alg=HyperOptSearch(),
        num_samples=2,
        config=config)


def test_nevergrad(trainable, config):
    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        search_alg=NevergradSearch(optimizer='PSO', budget=2),
        num_samples=2,
        config=config)


def test_trust_region(trainable, config):
    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        search_alg=TrustRegionSearcher(),
        num_samples=2,
        config=config)
