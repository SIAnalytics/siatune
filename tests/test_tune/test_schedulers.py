from tempfile import TemporaryDirectory

import pytest
from mmcv import Config
from ray.air import session
from ray.tune.error import TuneError

from siatune.tune import Tuner
from siatune.tune.schedulers import TRIAL_SCHEDULERS, build_scheduler


def test_build_scheduler():

    @TRIAL_SCHEDULERS.register_module()
    class TestScheduler:
        pass

    cfg = dict(type='TestScheduler')
    assert isinstance(build_scheduler(cfg), TestScheduler)


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


def test_asynchb(trainable, param_space):
    with TemporaryDirectory() as tmpdir:
        Tuner(
            trainable,
            tmpdir,
            param_space=param_space,
            tune_cfg=dict(metric='loss', mode='min', num_samples=2),
            trial_scheduler=dict(
                type='AsyncHyperBandScheduler',
                time_attr='training_iteration'),
            cfg=Config()).tune()


def test_hb(trainable, param_space):
    with TemporaryDirectory() as tmpdir:
        Tuner(
            trainable,
            tmpdir,
            param_space=param_space,
            tune_cfg=dict(metric='loss', mode='min', num_samples=2),
            trial_scheduler=dict(
                type='HyperBandScheduler', time_attr='training_iteration'),
            cfg=Config()).tune()


def test_bohb(trainable, param_space):
    with pytest.raises(TuneError):
        # If TuneBOHB is not used, an AttributeError will be raised
        with TemporaryDirectory() as tmpdir:
            Tuner(
                trainable,
                tmpdir,
                param_space=param_space,
                tune_cfg=dict(metric='loss', mode='min', num_samples=2),
                searcher=None,
                trial_scheduler=dict(
                    type='HyperBandForBOHB', time_attr='training_iteration'),
                cfg=Config()).tune()

    with TemporaryDirectory() as tmpdir:
        Tuner(
            trainable,
            tmpdir,
            param_space=param_space,
            tune_cfg=dict(metric='loss', mode='min', num_samples=2),
            searcher=dict(type='TuneBOHB'),
            trial_scheduler=dict(
                type='HyperBandForBOHB', time_attr='training_iteration'),
            cfg=Config()).tune()


def test_median(trainable, param_space):
    with TemporaryDirectory() as tmpdir:
        Tuner(
            trainable,
            tmpdir,
            param_space=param_space,
            tune_cfg=dict(metric='loss', mode='min', num_samples=2),
            searcher=dict(type='TuneBOHB'),
            trial_scheduler=dict(
                type='MedianStoppingRule', time_attr='time_total_s'),
            cfg=Config()).tune()
