import pytest
from ray import tune

from siatune.ray.spaces import Choice, GridSearch, SampleFrom, build_space
from siatune.utils import ImmutableContainer


def test_build_space():
    space = dict(
        a=dict(type='Uniform', lower=0.0, upper=1.0),
        b=dict(type='Randn', mean=0.0, sd=1.0))
    space = build_space(space)
    assert str(space.get('a').get_sampler()) == 'Uniform'
    assert str(space.get('b').get_sampler()) == 'Normal'


def test_choice():

    def is_in(config):
        assert config['test'] in [0, 1, 2]

    choice = Choice(categories=[0, 1, 2])
    tune.run(is_in, config=dict(test=choice.space))

    # with alias
    def is_immutable(config):
        assert isinstance(config['test'], ImmutableContainer)
        assert config['test'].data in [True, False]
        assert config['test'].alias in ['T', 'F']

    with pytest.raises(AssertionError):
        choice = Choice(categories=[True, False], alias=['TF'])
    choice = Choice(categories=[True, False], alias=['T', 'F'])
    tune.run(is_immutable, config=dict(test=choice.space))


def test_grid_search():

    def is_in(config):
        assert config['test1'] in [0, 1, 2]
        assert config['test2'] in [3, 4, 5]

    grid1 = GridSearch(values=[0, 1, 2])
    grid2 = GridSearch(values=[3, 4, 5])
    tune.run(is_in, config=dict(test1=grid1.space, test2=grid2.space))

    # with alias
    def is_immutable(config):
        for test in ['test1', 'test2']:
            assert isinstance(config[test], ImmutableContainer)
            assert config[test].data in [True, False]
            assert config[test].alias in ['T', 'F']

    with pytest.raises(AssertionError):
        grid1 = GridSearch(values=[True, False], alias=['TF'])
    grid1 = GridSearch(values=[True, False], alias=['T', 'F'])
    grid2 = GridSearch(values=[False, True], alias=['F', 'T'])
    tune.run(is_immutable, config=dict(test1=grid1.space, test2=grid2.space))


def test_sample_from():

    def is_eq(config):
        assert config['test'] == config['base']**2

    with pytest.raises(AssertionError):
        sample_from = SampleFrom('wrong expression')
    sample_from = SampleFrom('lambda spec: spec.config.base ** 2')
    tune.run(is_eq, config=dict(base=10, test=sample_from.space))

    sample_from = SampleFrom(lambda spec: spec.config.base**2)
    tune.run(is_eq, config=dict(base=10, test=sample_from.space))
