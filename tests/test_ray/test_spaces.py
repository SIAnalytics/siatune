from mmcv.utils import Config
from ray import tune

from mmtune.ray.spaces import BaseSpace, Choice, Constant, build_space


def test_base_space():
    assert BaseSpace().space is None


def test_build_space():
    space = dict(
        a=Config(dict(type='Uniform', low=0.0, high=1.0)),
        b=Config(dict(type='Randn', mean=0.0, sd=1.0)))
    space = build_space(space)
    assert space == dict(a=tune.uniform(0.0, 1.0), b=tune.randn(0.0, 1.0))


def test_choice():

    def objective(config):
        tune.report(test=config['test'])

    tune.run(
        objective,
        config=dict(
            test=Choice(
                categories=[1, 2, 3],
                alias=['one', 'two', 'three'],
                use_container=False)))
    assert tune.session.get_session()._queue.get()['test'] in [1, 2, 3]


def test_constant():

    def objective(config):
        tune.report(test=config['test'])

    tune.run(
        objective, config=dict(test=Constant(value=-1, use_container=False)))

    assert tune.session.get_session()._queue.get()['test'] == -1
