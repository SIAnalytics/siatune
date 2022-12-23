import random
import tempfile

from mmcv import Config
from ray.air import session

from siatune.codebase import TASKS
from siatune.tune import Tuner


@TASKS.register_module()
class TestTask:

    def create_trainable(self):
        return lambda cfg: session.report(dict(test=random.random()))


def test_tuner():

    with tempfile.TemporaryDirectory() as tmpdir:
        tune_cfg = Config(
            dict(
                task=dict(type='TestTask'),
                work_dir=tmpdir,
                tune_config=dict(metric='test', mode='min', num_samples=1)))
        tuner = Tuner.from_cfg(tune_cfg)
        tuner.tune()
