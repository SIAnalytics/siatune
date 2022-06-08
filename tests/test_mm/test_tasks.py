import argparse
from unittest.mock import patch

import numpy as np
import pytest
import torch
from mmcv.utils import Config
from ray import tune

from mmtune.mm.tasks import (TASKS, BaseTask, BlackBoxTask, MMClassification,
                             MMDetection, MMSegmentation, MMTrainBasedTask,
                             Sphere, build_task_processor)

_session = dict()


def report_to_session(*args, **kwargs):
    _session = get_session()
    _session = kwargs.copy()
    for arg in args:
        if isinstance(arg, dict):
            _session.update(arg)


def get_session():
    global _session
    return _session


@patch('ray.tune.report', side_effect=report_to_session)
def test_base_task():
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


@patch.object(MMSegmentation, 'train_model', return_value=None)
@patch.object(MMSegmentation, 'build_dataset')
def test_mmseg(mock_build_dataset, mock_train_model):
    mock_build_dataset.return_value.CLASSES = ['a', 'b', 'c']

    config_path = 'configs/mmseg/pspnet/pspnet_r18-d8_4x4_512x512_80k_potsdam.py'  # noqa

    task = MMSegmentation()
    task.set_args([config_path])
    task.run(args=task.args)


@patch.object(MMDetection, 'train_model', return_value=None)
@patch.object(MMDetection, 'build_dataset')
def test_mmdet(mock_build_dataset, mock_train_model):
    mock_build_dataset.return_value.CLASSES = ['a', 'b', 'c']

    config_path = 'configs/mmdet/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

    task = MMDetection()
    task.set_args([config_path])
    task.run(args=task.args)


@patch.object(MMClassification, 'train_model', return_value=None)
@patch.object(MMClassification, 'build_dataset')
def test_mmcls(mock_build_dataset, mock_train_model):
    mock_build_dataset.return_value.CLASSES = ['a', 'b', 'c']

    config_path = 'configs/mmcls/resnet/resnet18_8xb16_cifar10.py'

    task = MMClassification()
    task.set_args([config_path])
    task.run(args=task.args)


@patch('ray.tune.report', side_effect=report_to_session)
def test_mm_train_based_task():
    with pytest.raises(TypeError):
        MMTrainBasedTask()

    class TestTask(MMTrainBasedTask):

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
                    np.random.seed(0)
                    self._x = np.randn(num_points)
                    self._y = 2 * self._x + 1
                    self.num_points = num_points

                def __getitem__(self, index):
                    return torch.FloatTensor(
                        self._x[index]), torch.FloatTensor(self._y[index])

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
                tune.report(loss=total_loss / (batch_idx + 1))
            self.train_model(model, dataset, cfg.train)

        def run(self, *, searched_cfg, **kwargs):
            cfg = searched_cfg.get('cfg')
            model = self.build_model(cfg.model)
            dataset = self.build_dataset(cfg.dataset)
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
                batch_size=4,
                num_epochs=4,
            )))

    task = TestTask()
    task.context_aware_run(searched_cfg=dict(cfg=cfg))
    assert 'loss' in get_session()
    tune.run(
        task.create_trainable(num_gpus_per_worker=0), config=dict(cfg=cfg))


@patch.object(Config, 'fromfile')
@patch('ray.tune.report', side_effect=report_to_session)
def test_sphere(mock_fromfile):
    mock_fromfile.return_value = Config(dict(
        _variable0=-1,
        _variable1=-1,
    ))
    args = argparse.Namespace(config='')
    task = Sphere()  # noqa
    task.run(args)

    assert get_session().get('result') == 1
