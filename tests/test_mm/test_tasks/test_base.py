import argparse

import pytest
import ray
from ray import tune
from ray.tune.result_grid import ResultGrid

from siatune.mm.tasks import TASKS, BaseTask, build_task_processor


@pytest.fixture
def init_ray():
    if ray.is_initialized():
        ray.shutdown()
    return ray.init(num_cpus=1)


def test_base_task(init_ray):
    with pytest.raises(TypeError):
        BaseTask()

    class TestTask(BaseTask):

        def parse_args(self, args):
            parser = argparse.ArgumentParser()
            parser.add_argument('test')
            return parser.parse_args(args)

        def run(self, args):
            tune.report(test=args.test)

        def create_trainable(self):
            return self.context_aware_run

    class TestRewriter:

        def __call__(self, context):
            args = context.pop('args')
            args.test = 'success'
            return dict(args=args)

    task = TestTask(rewriters=[TestRewriter()])
    task.set_args(['default'])
    assert task.args == argparse.Namespace(test='default')

    trainable = task.create_trainable()
    results = ResultGrid(tune.run(trainable, config={}))
    assert results[0].metrics['test'] == 'success'


def test_build_task_processor():

    @TASKS.register_module()
    class TestTask(BaseTask):

        def parse_args(self, args):
            pass

        def run(self, args):
            pass

        def create_trainable(self):
            pass

    task = build_task_processor(dict(type='TestTask', rewriters=[]))
    assert isinstance(task, (BaseTask, TestTask))
