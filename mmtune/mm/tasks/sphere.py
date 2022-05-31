import numpy as np
import ray
from mmcv.utils import Config

from .blackbox import BlackBoxTask
from .builder import TASKS


@TASKS.register_module()
class Sphere(BlackBoxTask):

    def run(self, *, args):
        args = self.args
        cfg = Config.fromfile(args.config)

        inputs = []
        for k, v in cfg.items():
            if k.startswith('_variable'):
                inputs.append(v)
        inputs = np.array(inputs)

        ray.tune.report(result=float(inputs.dot(inputs)))
