# Copyright (c) SI-Analytics. All rights reserved.
#
# Brought from https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/functions/corefuncs.py # noqa E501
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from math import exp, sqrt, tanh

import numpy as np
import ray
from mmengine.config import Config

from .blackbox import BlackBoxTask
from .builder import TASKS


def _styblinksitang(x: np.ndarray, noise: float) -> float:
    """Classical function for testing noisy optimization.

    Args:
        x (np.ndarray): input vector.
        noise (float): noise level.
    Returns:
        float: output.
    """
    x2 = x**2
    val = x2.dot(x2) + np.sum(5 * x - 16 * x2)
    # return a positive value for maximization
    return float(39.16599 * len(x) + 0.5 * val +  # noqa W504
                 noise * np.random.normal(size=val.shape))


def _step(s: float, eps: float = 1e-8) -> float:
    """Step function.

    Args:
        s (float): input.
        eps (float): small number to avoid numerical errors.
    Returns:
        float: output.
    """
    if s == 0:
        s += eps
    return np.nan_to_num(np.exp(np.round(np.log(s))))


@TASKS.register_module()
class ContinuousTestFunction(BlackBoxTask):
    """Continuous test function."""

    @staticmethod
    def delayedsphere(x: np.ndarray) -> float:
        """Delayed sphere function.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return float(np.sum(x**2))

    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """The most classical continuous optimization testbed.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        assert x.ndim == 1
        return float(x.dot(x))

    # TODO: change translated the sphere function as partial function
    @staticmethod
    def sphere1(x: np.ndarray) -> float:
        """Translated sphere function.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return ContinuousTestFunction.sphere(x - 1.0)

    @staticmethod
    def sphere2(x: np.ndarray) -> float:
        """A bit more translated sphere function.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return ContinuousTestFunction.sphere(x - 2.0)

    @staticmethod
    def sphere4(x: np.ndarray) -> float:
        """Even more translated sphere function.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return ContinuousTestFunction.sphere(x - 4.0)

    @staticmethod
    def maxdeceptive(x: np.ndarray) -> float:
        """Max deceptive function.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        dec = 3 * x**2 - (2 / (3**(x - 2)**2 + 0.1))
        return float(np.max(dec))

    @staticmethod
    def sumdeceptive(x: np.ndarray) -> float:
        """Sum deceptive function.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        dec = 3 * x**2 - (2 / (3**(x - 2)**2 + 0.1))
        return float(np.sum(dec))

    @staticmethod
    def altcigar(x: np.ndarray) -> float:
        """Similar to cigar, but variables in inverse order.

        E.g. for pointing out algorithms not invariant to the order of
        variables.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return float(
            x[-1])**2 + 1000000.0 * ContinuousTestFunction.sphere(x[:-1])

    @staticmethod
    def discus(x: np.ndarray) -> float:
        """Only one variable is very penalized.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return ContinuousTestFunction.sphere(
            x[1:]) + 1000000.0 * float(x[0])**2

    @staticmethod
    def cigar(x: np.ndarray) -> float:
        """Classical example of ill conditioned function.

        The other classical example is ellipsoid.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return float(
            x[0])**2 + 1000000.0 * ContinuousTestFunction.sphere(x[1:])

    @staticmethod
    def bentcigar(x: np.ndarray) -> float:
        """Classical example of ill conditioned function, but bent.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        y = np.asarray([
            x[i]**(
                1 + 0.5 * np.sqrt(x[i]) *  # noqa W504
                (i - 1)(len(x) - 1)) if x[i] > 0.0 else x[i]
            for i in range(len(x))
        ])
        return float(
            y[0])**2 + 1000000.0 * ContinuousTestFunction.sphere(y[1:])

    @staticmethod
    def multipeak(x: np.ndarray) -> float:
        """Inspired by M.

        Gallagher's Gaussian peaks function.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        v = 10000.0
        for a in range(101):
            x_ = np.asarray([np.cos(a + np.sqrt(i)) for i in range(len(x))])
            v = min(v,
                    a / 101.0 + np.exp(ContinuousTestFunction.sphere(x - x_)))
        return v

    @staticmethod
    def altellipsoid(x: np.ndarray) -> float:
        """Similar to Ellipsoid, but variables in inverse order.

        E.g. for pointing out algorithms not invariant to the order of
        variables.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return ContinuousTestFunction.ellipsoid(x[::-1])

    @staticmethod
    def stepellipsoid(x: np.ndarray) -> float:
        """Classical example of ill conditioned function.

        But we add a 'step', i.e. we set the gradient to zero everywhere.
        Compared to some existing testbeds, we decided to have infinitely many
        steps.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        dim = x.size
        weights = 10**np.linspace(0, 6, dim)
        return float(_step(weights.dot(x**2)))

    @staticmethod
    def ellipsoid(x: np.ndarray) -> float:
        """Classical example of ill conditioned function.

        The other classical example is cigar.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        dim = x.size
        weights = 10**np.linspace(0, 6, dim)
        return float(weights.dot(x**2))

    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """Classical multimodal function.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        cosi = float(np.sum(np.cos(2 * np.pi * x)))
        return float(10 * (len(x) - cosi) + ContinuousTestFunction.sphere(x))

    @staticmethod
    def bucherastrigin(x: np.ndarray) -> float:
        """Classical multimodal function.

        No box-constraint penalization here.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        s = np.asarray([
            x[i] * (10 if x[i] > 0.0 and i % 2 else 1) *  # noqa W504
            (10**((i - 1) / (2 * (len(x) - 1)))) for i in range(len(x))
        ])
        cosi = float(np.sum(np.cos(2 * np.pi * s)))
        return float(10 * (len(x) - cosi) + ContinuousTestFunction.sphere(s))

    @staticmethod
    def doublelinearslope(x: np.ndarray) -> float:
        """We decided to use two linear slopes rather than having a constraint
        artificially added for not having the optimum at infinity.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return float(np.abs(np.sum(x)))

    @staticmethod
    def stepdoublelinearslope(x: np.ndarray) -> float:
        """Double Lenear Slope with step.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return _step(np.abs(np.sum(x)))

    @staticmethod
    def hm(x: np.ndarray, eps: float = 1e-8) -> float:
        """New multimodal function (proposed for Nevergrad).

        Args:
            x (np.ndarray): input vector.
            eps (float): small number to avoid numerical errors.
        Returns:
            float: output.
        """
        if not np.all(x):
            x += eps
        return float((x**2).dot(1.1 + np.cos(1.0 / x)))

    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """Rosenbrock function is a non-convex function, introduced by Howard
        H. Rosenbrock in 1960.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        x_m_1 = x[:-1] - 1
        x_diff = x[:-1]**2 - x[1:]
        return float(100 * x_diff.dot(x_diff) + x_m_1.dot(x_m_1))

    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """Ackley function is a non-convex function used as a performance test
        problem for optimization algorithms. It was proposed by David Ackley in
        his 1987 PhD dissertation.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        dim = x.size
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        return (
            -20.0 * exp(-0.2 * sqrt(ContinuousTestFunction.sphere(x) / dim))
            -  # noqa W504
            exp(sum_cos / dim) + 20 + exp(1))

    @staticmethod
    def schwefel_1_2(x: np.ndarray) -> float:
        """Schwefel's function is deceptive in that the global minimum is
        geometrically distant, over the parameter space, from the next best
        local minima. Therefore, the search algorithms are potentially prone to
        convergence in the wrong direction.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        cx = np.cumsum(x)
        return ContinuousTestFunction.sphere(cx)

    @staticmethod
    def griewank(x: np.ndarray) -> float:
        """Multimodal function, often used in Bayesian optimization.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        part1 = ContinuousTestFunction.sphere(x)
        part2 = np.prod(np.cos(x / np.sqrt(1 + np.arange(len(x)))))
        return 1 + (float(part1) / 4000.0) - float(part2)

    @staticmethod
    def deceptiveillcond(x: np.ndarray) -> float:
        """An extreme ill conditioned functions.

        Most algorithms fail on this. The condition number increases to
        infinity as we get closer to the optimum.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        assert len(x) >= 2
        return float(
            max(
                np.abs(np.arctan(x[1] / x[0])),
                np.sqrt(x[0]**2.0 + x[1]**2.0),
                1.0 if x[0] > 0 else 0.0,
            ) if x[0] != 0.0 else float('inf'))

    @staticmethod
    def deceptivepath(x: np.ndarray) -> float:
        """A function which needs following a long path.

        Most algorithms fail on this. The path becomes thiner as we get closer
        to the optimum.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        assert len(x) >= 2
        distance = np.sqrt(x[0]**2 + x[1]**2)
        if distance == 0.0:
            return 0.0
        angle = np.arctan(x[0] / x[1]) if x[1] != 0.0 else np.pi / 2.0
        invdistance = (1.0 / distance) if distance > 0.0 else 0.0
        if np.abs(np.cos(invdistance) - angle) > 0.1:
            return 1.0
        return float(distance)

    @staticmethod
    def deceptivemultimodal(x: np.ndarray) -> float:
        """Infinitely many local optima, as we get closer to the optimum.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        assert len(x) >= 2
        distance = np.sqrt(x[0]**2 + x[1]**2)
        if distance == 0.0:
            return 0.0
        angle = np.arctan(x[0] / x[1]) if x[1] != 0.0 else np.pi / 2.0
        invdistance = int(1.0 / distance) if distance > 0.0 else 0.0
        if np.abs(np.cos(invdistance) - angle) > 0.1:
            return 1.0
        return float(distance)

    @staticmethod
    def lunacek(x: np.ndarray) -> float:
        """Known as the bior double-Rastrigin function, is a hybrid function
        consisting of a Rastrigin and a double-sphere part.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        problemDimensions = len(x)
        s = 1.0 - (1.0 / (2.0 * np.sqrt(problemDimensions + 20.0) - 8.2))
        mu1 = 2.5
        mu2 = -np.sqrt(abs((mu1**2 - 1.0) / s))
        firstSum = 0.0
        secondSum = 0.0
        thirdSum = 0.0
        for i in range(problemDimensions):
            firstSum += (x[i] - mu1)**2
            secondSum += (x[i] - mu2)**2
            thirdSum += 1.0 - np.cos(2 * np.pi * (x[i] - mu1))
        return min(firstSum,
                   1.0 * problemDimensions + secondSum) + 10 * thirdSum

    @staticmethod
    def genzcornerpeak(x: np.ndarray) -> float:
        """One of the Genz functions, originally used in integration, tested in
        optim because why not.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        value = float(1 + np.mean(np.tanh(x)))
        if value == 0:
            return float('inf')
        return value**(-len(x) - 1)

    @staticmethod
    def minusgenzcornerpeak(x: np.ndarray) -> float:
        """One of the Genz functions, originally used in integration, tested in
        optim because why not.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return -ContinuousTestFunction.genzcornerpeak(x)

    @staticmethod
    def genzgaussianpeakintegral(x: np.ndarray) -> float:
        """One of the Genz functions, originally used in integration, tested in
        optim because why not.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return exp(-ContinuousTestFunction.sphere(x) / 4.0)

    @staticmethod
    def minusgenzgaussianpeakintegral(x: np.ndarray) -> float:
        """One of the Genz functions, originally used in integration, tested in
        optim because why not.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return -ContinuousTestFunction.genzgaussianpeakintegral(x)

    @staticmethod
    def slope(x: np.ndarray) -> float:
        """The summation of the vector.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return sum(x)

    @staticmethod
    def linear(x: np.ndarray) -> float:
        """The hyperbolic tangent of first element.

        Args:
            x (np.ndarray): input vector.

        Returns:
            float: output.
        """
        return tanh(x[0])

    @staticmethod
    def st0(x: np.ndarray) -> float:
        """Styblinksitang function with 0 noise.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return _styblinksitang(x, 0)

    @staticmethod
    def st1(x: np.ndarray) -> float:
        """Styblinksitang function with noise 1.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return _styblinksitang(x, 1)

    @staticmethod
    def st10(x: np.ndarray) -> float:
        """Styblinksitang function with noise 10.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return _styblinksitang(x, 10)

    @staticmethod
    def st100(x: np.ndarray) -> float:
        """Styblinksitang function with noise 100.

        Args:
            x (np.ndarray): input vector.
        Returns:
            float: output.
        """
        return _styblinksitang(x, 100)

    def run(self, *, args: argparse.Namespace, **kwargs) -> None:
        """Run the task.

        Args:
            args (argparse.Namespace): The arguments.
        """
        cfg = Config.fromfile(args.config)
        func = getattr(self, cfg.get('func', 'sphere'))
        inputs = []
        for k, v in cfg.items():
            if k.startswith('_variable'):
                inputs.append(v)
        inputs = np.array(inputs)

        ray.tune.report(result=func(inputs))
