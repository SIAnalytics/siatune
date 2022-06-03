import argparse

import numpy as np
import ray
from mmcv.utils import Config

from .blackbox import BlackBoxTask
from .builder import TASKS


@TASKS.register_module()
class Sphere(BlackBoxTask):
    """Test function for continuous evaluation.

    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """

    def run(self, *, args: argparse.Namespace, **kwargs) -> None:
        """Run the task.

        Args:
            args (argparse.Namespace): The arguments.
        """
        cfg = Config.fromfile(args.config)

        inputs = []
        for k, v in cfg.items():
            if k.startswith('_variable'):
                inputs.append(v)
        inputs = np.array(inputs)

        ray.tune.report(result=float(inputs.dot(inputs)))
