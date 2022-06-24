from ray import tune

from mmtune.ray.stoppers import (STOPPERS, DictionaryStopper,
                                 EarlyDroppingStopper, build_stopper)


def test_build_stopper():

    @STOPPERS.register_module()
    class TestStopper:
        pass

    assert isinstance(build_stopper({'type': 'TestStopper'}), TestStopper)


def test_dictionarystopper():

    class TestTrainable(tune.Trainable):

        def setup(self, config):
            self._itr = 0

        def step(self):
            self._itr += 1
            if self._itr > 10:
                raise Exception('Test')
            return dict(result=-1)

    stopper = DictionaryStopper(training_iteration=10)
    tune.run(TestTrainable, config={}, stop=stopper)


def test_earlydroppingstopper():

    class TestTrainable(tune.Trainable):

        def setup(self, config):
            self._itr = 0

        def step(self):
            self._itr += 1
            if self._itr > 1:
                raise Exception('Test')
            return dict(result=0)

    stopper = EarlyDroppingStopper(
        metric='result', mode='max', metric_threshold=0.5)
    tune.run(TestTrainable, config={}, stop=stopper)
