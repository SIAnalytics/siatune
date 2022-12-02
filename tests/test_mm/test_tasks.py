import argparse
import os
from unittest.mock import patch

import pytest
import torch
from mmcv.utils import Config
from ray import tune
from ray.air import session

from siatune.mm.tasks import (TASKS, BaseTask, BlackBoxTask,
                              ContinuousTestFunction, DiscreteTestFunction,
                              MMClassification, MMDetection, MMSegmentation,
                              MMTrainBasedTask, build_task_processor)
from siatune.utils.config import dump_cfg

_session = dict()


def report_to_session(*args, **kwargs):
    _session = get_session()
    _session.update(kwargs)
    for arg in args:
        if isinstance(arg, dict):
            _session.update(arg)


def get_session():
    global _session
    return _session


@patch('ray.tune.report', side_effect=report_to_session)
def test_base_task(mock_report):
    with pytest.raises(TypeError):
        BaseTask()

    class TestRewriter:

        def __call__(self, context):
            context.get('args').test = -1
            return context

    class TestTask(BaseTask):

        def parse_args(self, *args, **kwargs):
            return argparse.Namespace(test=1)

        def run(self, *, args, **kwargs):
            tune.report(test=args.test)
            return args.test

        def create_trainable(self):
            return self.context_aware_run

    task = TestTask([TestRewriter()])
    task.set_args('')
    assert task.args == argparse.Namespace(test=1)
    assert isinstance(task.rewriters, list)
    task.context_aware_run({})
    assert get_session().get('test') == -1

    tune.run(task.create_trainable(), config={})


def test_black_box_task():
    with pytest.raises(TypeError):
        BlackBoxTask()

    class TestTask(BlackBoxTask):

        def run(self, *args, **kwargs):
            tune.report(test=1)

    task = TestTask()
    task.set_args('')
    assert task.args == argparse.Namespace()
    tune.run(task.create_trainable(), config={})


def test_build_task_processor():

    class TestTaks(BaseTask):

        def parse_args(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            pass

        def create_trainable(self, *args, **kwargs):
            pass

    TASKS.register_module(TestTaks)
    assert isinstance(build_task_processor(dict(type='TestTaks')), TestTaks)


@patch('ray.tune.report', side_effect=report_to_session)
def test_continuous_test_function(mock_report):
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
        assert isinstance(get_session().get('result'), float)


@patch('ray.tune.report', side_effect=report_to_session)
def test_discrete_test_function(mock_report):
    func = DiscreteTestFunction()

    predefined_discrete_funcs = ['onemax', 'leadingones', 'jump']
    for func_name in predefined_discrete_funcs:
        dump_cfg(
            Config(dict(func=func_name, _variable0=0.0, _variable1=0.0)),
            'test.py')
        args = argparse.Namespace(config='test.py')
        func.run(args=args)
        assert isinstance(get_session().get('result'), float)


@patch.object(MMSegmentation, 'train_model')
@patch.object(MMSegmentation, 'build_model')
@patch.object(MMSegmentation, 'build_dataset')
def test_mmseg(*not_used):
    os.environ['LOCAL_RANK'] = '0'

    task = MMSegmentation()
    task.set_args(['tests/data/config.py'])
    task.run(args=task.args)


@patch.object(MMDetection, 'train_model')
@patch.object(MMDetection, 'build_model')
@patch.object(MMDetection, 'build_dataset')
def test_mmdet(*not_used):
    os.environ['LOCAL_RANK'] = '0'

    task = MMDetection()
    task.set_args(['tests/data/config.py'])
    task.run(args=task.args)


@patch.object(MMClassification, 'train_model')
@patch.object(MMClassification, 'build_model')
@patch.object(MMClassification, 'build_dataset')
def test_mmcls(*not_used):
    os.environ['LOCAL_RANK'] = '0'

    task = MMClassification()
    task.set_args(['tests/data/config.py'])
    task.run(args=task.args)


@patch('ray.air.session.report', side_effect=report_to_session)
def test_mm_train_based_task(mock_report):
    with pytest.raises(TypeError):
        MMTrainBasedTask()

    class TestTask(MMTrainBasedTask):

        def parse_args(self, args):
            parser = argparse.ArgumentParser()
            return parser.parse_args(args)

        def build_model(self, cfg):

            class Regression(torch.nn.Module):

                def __init__(self, input_dim, output_dim):
                    super().__init__()
                    self.linear = torch.nn.Linear(input_dim, output_dim)

                def forward(self, x):
                    return self.linear(x)

            return Regression(cfg.input_dim, cfg.output_dim)

        def build_dataset(self, cfg):

            class Dataset(torch.utils.data.Dataset):

                def __init__(self, num_points):
                    torch.manual_seed(0)
                    self._x = torch.randn(num_points, 1)
                    self._y = 2 * self._x + 1
                    self.num_points = num_points

                def __getitem__(self, index):
                    return self._x[index], self._y[index]

                def __len__(self):
                    return self.num_points

            return Dataset(cfg.num_points)

        def train_model(self, model, dataset, cfg):
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=cfg.batch_size)
            for _ in range(cfg.num_epochs):
                total_loss = 0.
                for batch_idx, (data, target) in enumerate(data_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                session.report(loss=total_loss / (batch_idx + 1))

        def run(self, *, searched_cfg, **kwargs):
            cfg = searched_cfg.get('cfg')
            model = self.build_model(cfg.model)
            dataset = self.build_dataset(cfg.data)
            self.train_model(model, dataset, cfg.train)

    cfg = Config(
        dict(
            model=dict(
                input_dim=1,
                output_dim=1,
            ),
            data=dict(num_points=128, ),
            train=dict(
                lr=0.1,
                batch_size=32,
                num_epochs=4,
            )))

    task = TestTask()
    task.set_resource(1, 0, 1)
    task.context_aware_run(searched_cfg=dict(cfg=cfg))
    assert 'loss' in get_session()

    trainable = task.create_trainable()
    tune.Tuner(trainable).fit()
