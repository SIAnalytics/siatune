import argparse
from unittest.mock import patch

from mmcv.utils import Config

from siatune.core.tasks import ContinuousTestFunction, DiscreteTestFunction
from siatune.utils.config import dump_cfg

session = dict()


def report_to_session(**kwargs):
    session.update(kwargs)


@patch('ray.tune.report', side_effect=report_to_session)
def test_continuous_test_function(*not_used):
    func = ContinuousTestFunction()
    predefined_cont_funcs = [
        'delayedsphere',
        'sphere',
        'sphere1',
        'sphere2',
        'sphere4',
        'maxdeceptive',
        'sumdeceptive',
        'altcigar',
        'discus',
        'cigar',
        'bentcigar',
        'multipeak',
        'altellipsoid',
        'stepellipsoid',
        'ellipsoid',
        'rastrigin',
        'bucherastrigin',
        'doublelinearslope',
        'stepdoublelinearslope',
        'hm',
        'rosenbrock',
        'ackley',
        'schwefel_1_2',
        'griewank',
        'deceptiveillcond',
        'deceptivepath',
        'deceptivemultimodal',
        'lunacek',
        'genzcornerpeak',
        'minusgenzcornerpeak',
        'genzgaussianpeakintegral',
        'minusgenzgaussianpeakintegral',
        'slope',
        'linear',
        'st0',
        'st1',
        'st10',
        'st100',
    ]

    for func_name in predefined_cont_funcs:
        dump_cfg(
            Config(dict(func=func_name, _variable0=0.0, _variable1=0.0)),
            'test.py')
        args = argparse.Namespace(config='test.py')
        func.run(args=args)
        assert isinstance(session['result'], float)


@patch('ray.tune.report', side_effect=report_to_session)
def test_discrete_test_function(*not_used):
    func = DiscreteTestFunction()

    predefined_discrete_funcs = ['onemax', 'leadingones', 'jump']
    for func_name in predefined_discrete_funcs:
        dump_cfg(
            Config(dict(func=func_name, _variable0=0.0, _variable1=0.0)),
            'test.py')
        args = argparse.Namespace(config='test.py')
        func.run(args=args)
        assert isinstance(session['result'], float)
