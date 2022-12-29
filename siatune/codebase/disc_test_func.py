# Copyright (c) SI-Analytics. All rights reserved.
#
# Brought from https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/functions/corefuncs.py # noqa E501
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import numpy as np
import ray
from mmengine.config import Config

from .blackbox import BlackBoxTask
from .builder import TASKS


@TASKS.register_module()
class DiscreteTestFunction(BlackBoxTask):
    """Test functions for discrete optimization.

    Returns a classical discrete function for test, in the domain
    {0,1,...,arity-1}^d
    """

    @staticmethod
    def onemax(x: np.ndarray, arity: int) -> float:
        """onemax(x) is the most classical case of discrete functions, adapted
        to minimization. It is originally designed for lists of bits. It just
        counts the number of 1,

        and returns len(x) - number of ones.
        However, the present function perturbs the location of the optimum,
        so that tests can not be easily biased by a wrong initialization.
        So the optimum,
        instead of being located at (1,1,...,1),
        is located at (0,1,2,...,arity-1,0,1,2,...).

        Args:
            x (np.ndarray): input vector
            arity (int): The number of categories a discrete variable can take

        Returns:
            float: output of the function.
        """
        diff = np.round(x) - (np.arange(len(x)) % arity)
        return float(np.sum(diff != 0))

    @staticmethod
    def leadingones(x: np.ndarray, arity: int) -> float:
        """leadingones is the second most classical discrete function, adapted
        for minimization. Before perturbation of the location of the optimum as
        above,

        it returns len(x) - number of initial 1. I.e.
        leadingones([0 1 1 1]) = 4,
        leadingones([1 1 1 1]) = 0,
        leadingones([1 0 0 0]) = 3.

        Args:
            x (np.ndarray): input vector
            arity (int): The number of categories a discrete variable can take

        Returns:
            float: output of the function.
        """
        diff = np.round(x) - (np.arange(len(x)) % arity)
        nonzeros = np.nonzero(diff)[0]
        return float(len(x) - nonzeros[0] if nonzeros.size else 0)

    @staticmethod
    def jump(x: np.ndarray, arity: int) -> float:
        """A variant of the jump function based on the principle that local
        descent does not work.

        Args:
            x (np.ndarray): input vector
            arity (int): The number of categories a discrete variable can take

        Returns:
            float: output of the function.
        """
        n = len(x)
        m = n // 4
        o = n - DiscreteTestFunction.onemax(x, arity)
        if o == n or o <= n - m:
            return n - m - o
        return o

    def run(self, *, args: argparse.Namespace, **kwargs) -> None:
        """Run the task.

        Args:
            args (argparse.Namespace): The arguments.
        """
        cfg = Config.fromfile(args.config)
        arity = cfg.get('arity', 2)
        func = getattr(self, cfg.get('func', 'onemax'))

        inputs = []
        for k, v in cfg.items():
            if k.startswith('_variable'):
                inputs.append(v)
        inputs = np.array(inputs)
        return ray.tune.report(result=func(inputs, arity))
