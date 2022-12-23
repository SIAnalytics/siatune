import pytest
from ray import tune
from ray.tune.error import TuneError

from siatune.tune.schedulers import TRIAL_SCHEDULERS, build_scheduler
from siatune.tune.searchers import build_searcher


def test_build_schedulers():

    @TRIAL_SCHEDULERS.register_module()
    class TestScheduler:
        pass

    assert isinstance(
        build_scheduler(dict(type='TestScheduler')), TestScheduler)


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


def test_asynchb(trainable, config):
    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        scheduler=build_scheduler(
            dict(
                type='AsyncHyperBandScheduler',
                time_attr='training_iteration')),
        num_samples=2,
        config=config)


def test_hb(trainable, config):
    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        scheduler=build_scheduler(
            dict(type='HyperBandScheduler', time_attr='training_iteration')),
        num_samples=2,
        config=config)


def test_bohb(trainable, config):
    with pytest.raises(TuneError):
        # AttributeError
        tune.run(
            trainable,
            metric='mean_loss',
            mode='min',
            search_alg=None,
            scheduler=build_scheduler(
                dict(type='HyperBandForBOHB', time_attr='training_iteration')),
            num_samples=2,
            config=config)

    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        search_alg=build_searcher(dict(type='TuneBOHB')),
        scheduler=build_scheduler(
            dict(type='HyperBandForBOHB', time_attr='training_iteration')),
        num_samples=2,
        config=config)


def test_median(trainable, config):
    tune.run(
        trainable,
        metric='mean_loss',
        mode='min',
        scheduler=build_scheduler(
            dict(type='MedianStoppingRule', time_attr='time_total_s')),
        num_samples=2,
        config=config)
