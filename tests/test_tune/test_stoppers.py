from tempfile import TemporaryDirectory

from mmengine import Config
from ray.tune import Trainable

from siatune.tune import Tuner
from siatune.tune.stoppers import STOPPERS, build_stopper


def test_build_stopper():

    @STOPPERS.register_module()
    class TestStopper:
        pass

    cfg = dict(type='TestStopper')
    assert isinstance(build_stopper(cfg), TestStopper)


def test_dict():

    class TestTrainable(Trainable):

        def setup(self, config):
            self.iter = 0

        def step(self):
            self.iter += 1
            assert self.iter <= 10
            return dict(test='success')

    with TemporaryDirectory() as tmpdir:
        Tuner(
            TestTrainable,
            tmpdir,
            param_space=dict(),
            tune_cfg=dict(),
            stopper=dict(type='DictionaryStopper', training_iteration=10),
            cfg=Config()).tune()


def test_early_drop():

    class TestTrainable(Trainable):

        def setup(self, config):
            self.iter = 0

        def step(self):
            self.iter += 1
            assert self.iter <= 1
            return dict(test=0)

    with TemporaryDirectory() as tmpdir:
        Tuner(
            TestTrainable,
            tmpdir,
            param_space=dict(),
            tune_cfg=dict(),
            stopper=dict(
                type='EarlyDroppingStopper',
                metric='test',
                mode='max',
                metric_threshold=0.5),
            cfg=Config()).tune()
