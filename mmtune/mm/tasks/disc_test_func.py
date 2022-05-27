import numpy as np
import ray
from mmcv.utils import Config

from .blackbox import BlackBoxTask
from .builder import TASKS


@TASKS.register_module()
class DiscreteTestFunction(BlackBoxTask):

    @staticmethod
    def onemax(x: np.ndarray, arity: int) -> float:
        diff = np.round(x) - (np.arange(len(x)) % arity)
        return float(np.sum(diff != 0))

    @staticmethod
    def leadingones(x: np.ndarray, arity: int) -> float:
        diff = np.round(x) - (np.arange(len(x)) % arity)
        nonzeros = np.nonzero(diff)[0]
        return float(len(x) - nonzeros[0] if nonzeros.size else 0)

    @staticmethod
    def jump(x: np.ndarray, arity: int) -> float:
        n = len(x)
        m = n // 4
        o = n - DiscreteTestFunction.onemax(x, arity)
        if o == n or o <= n - m:
            return n - m - o
        return o

    def run(self, *args, **kwargs):
        args = kwargs['args']
        cfg = Config.fromfile(args.config)
        arity = cfg.get('arity', 2)
        func = getattr(self, cfg.get('func', 'leadingones'))

        inputs = []
        for k, v in cfg.items():
            if k.startswith('_variable'):
                inputs.append(v)
        inputs = np.array(inputs)
        return ray.tune.report(result=func(inputs, arity))
