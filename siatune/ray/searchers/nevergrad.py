# Copyright (c) SI-Analytics. All rights reserved.
from typing import Dict, List, Optional, Union

from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.nevergrad import NevergradSearch as _NevergradSearch

from .builder import SEARCHERS

try:
    import nevergrad as ng
    from nevergrad.optimization import Optimizer
    from nevergrad.optimization.base import ConfiguredOptimizer
    Parameter = ng.p.Parameter
    from nevergrad.optimization.optimizerlib import \
        registry as optimizer_registry
except ImportError:
    ng = None
    Optimizer = None
    ConfiguredOptimizer = None
    Parameter = None
    optimizer_registry = dict()


@SEARCHERS.register_module(force=True)
class NevergradSearch(_NevergradSearch):
    """Search with Nevergrad."""

    def __init__(self,
                 optimizer: str = 'OnePlusOne',
                 space: Optional[Union[Dict, Parameter]] = None,
                 metric: Optional[str] = None,
                 mode: Optional[str] = None,
                 points_to_evaluate: Optional[List[Dict]] = None,
                 num_workers: int = 1,
                 budget: Optional[int] = None,
                 **kwargs) -> None:
        """Initialize NevergradSearch.

        Args:
            optimizer (str): The optimizer. Defaults to 'OnePlusOne'.
            space (Optional[Union[Dict, Parameter]]):
                The space to search. Defaults to None.
            metric (Optional[str]):
                Performance evaluation metrics.. Defaults to None.
            mode (Optional[str]):
                Determines whether objective is
                minimizing or maximizing the metric attribute.
                Defaults to None.
            points_to_evaluate (Optional[List[Dict]]):
                Initial parameter suggestions to be run first.
                Defaults to None.
            num_workers (int):
                The number of evaluations
                which will be run in parallel at once.
                Defaults to 1.
            budget (Optional[int]):
                The number of allowed evaluations.
                Defaults to None.
        """
        assert optimizer in optimizer_registry, (
            f'{optimizer} is not registered')
        optimizer = optimizer_registry[optimizer]
        self._budget = budget
        self._num_workers = num_workers
        super(NevergradSearch, self).__init__(
            optimizer=optimizer,
            space=space,
            metric=metric,
            mode=mode,
            points_to_evaluate=points_to_evaluate,
            **kwargs)

    def _setup_nevergrad(self) -> None:
        """setup Nevergrad optimizer."""

        if self._opt_factory:
            self._nevergrad_opt = self._opt_factory(
                parametrization=self._space,
                budget=self._budget,
                num_workers=self._num_workers)

        # nevergrad.tell internally minimizes, so "max" => -1
        if self._mode == 'max':
            self._metric_op = -1.
        elif self._mode == 'min':
            self._metric_op = 1.

        if self._metric is None and self._mode:
            # If only a mode was passed, use anonymous metric
            self._metric = DEFAULT_METRIC

        if hasattr(self._nevergrad_opt, 'instrumentation'):  # added in v0.2.0
            if self._nevergrad_opt.instrumentation.kwargs:
                if self._nevergrad_opt.instrumentation.args:
                    raise ValueError(
                        'Instrumented optimizers should use kwargs only')
                if self._parameters is not None:
                    raise ValueError('Instrumented optimizers should provide '
                                     'None as parameter_names')
            else:
                if self._parameters is None:
                    raise ValueError('Non-instrumented optimizers should have '
                                     'a list of parameter_names')
                if len(self._nevergrad_opt.instrumentation.args) != 1:
                    raise ValueError(
                        'Instrumented optimizers should use kwargs only')
        if self._parameters is not None and (self._nevergrad_opt.dimension !=
                                             len(self._parameters)):
            raise ValueError('len(parameters_names) must match optimizer '
                             'dimension for non-instrumented optimizers')

        if self._points_to_evaluate:
            # Nevergrad is LIFO, so we add the points to evaluate in reverse
            # order.
            for i in range(len(self._points_to_evaluate) - 1, -1, -1):
                self._nevergrad_opt.suggest(self._points_to_evaluate[i])

    def add_evaluated_point(
        self,
        parameters: Dict,
        value: float,
        error: bool = False,
        pruned: bool = False,
        intermediate_values: Optional[List[float]] = None,
        trial_id: str = None,
    ) -> None:
        """Add evaluated point to Nevergrad optimizer.

        Args:
            parameters (Dict): The parameters.
            value (float): The value.
            error (bool):
                Whether the point was evaluated in error. Defaults to False.
            pruned (bool):
                Whether the point was pruned. Defaults to False.
            intermediate_values (Optional[List[float]]):
                The intermediate values. Defaults to None.
            trial_id (str): The trial id. Defaults to None.
        """

        candidate = self._nevergrad_opt.parametrization.spawn_child(
            new_value=parameters)

        self._nevergrad_opt.tell(candidate, self._metric_op * value)
