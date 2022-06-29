from ray import tune

from mmtune.ray.spaces import BaseSpace, Choice, build_space


def test_base_space():
    assert hasattr(BaseSpace(), 'space')


def test_build_space():
    space = dict(
        a=dict(type='Uniform', lower=0.0, upper=1.0),
        b=dict(type='Randn', mean=0.0, sd=1.0))
    space = build_space(space)
    assert str(space.get('a').get_sampler()) == 'Uniform'
    assert str(space.get('b').get_sampler()) == 'Normal'


def test_choice():

    def objective(config):
        assert config.get('test') in [1, 2, 3]
        tune.report(result=config['test'])

    tune.run(
        objective,
        config=dict(
            test=Choice(categories=[1, 2, 3], use_container=False).space))
