import numpy as np
import ray
from mmcv.utils import Config

from .builder import TASK


@TASK.register_module()
class Sphere:

    @staticmethod
    def run(*args, **kwargs):
        args = kwargs['args']
        cfg = Config.fromfile(args.config)

        inputs = []
        for k, v in cfg:
            if k.startswith('_variable'):
                inputs.append(v)
        inputs = np.array(inputs)

        ray.tune.report(result=float(inputs.dot(inputs)))
