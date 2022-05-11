from typing import Dict, List, Optional, Union

from ray.tune.result import DEFAULT_METRIC
from ray.tune.suggest import Searcher
from ray.tune.suggest.nevergrad import NevergradSearch as _NevergradSearch

from .builder import SEARCHERS

try:
    import nevergrad as ng
    from nevergrad.optimization import Optimizer
    from nevergrad.optimization.base import ConfiguredOptimizer
    Parameter = ng.p.Parameter
    from nevergrad.optimization.optimizerlib import registry as optimizer_registry
except ImportError:
    ng = None
    Optimizer = None
    ConfiguredOptimizer = None
    Parameter = ng.p.Parameter
    optimizer_registry = dict()


@SEARCHERS.register_module(force=True)
class NevergradSearch(_NevergradSearch, Searcher):

    def __init__(self,
                 optimizer: str = 'OnePlusOne',
                 space: Optional[Union[Dict, Parameter]] = None,
                 metric: Optional[str] = None,
                 mode: Optional[str] = None,
                 points_to_evaluate: Optional[List[Dict]] = None,
                 max_concurrent: Optional[int] = None,
                 budget: Optional[int] = None,
                 **kwargs):
        assert optimizer in optimizer_registry, f'{optimizer} is not registered'
        self._budget = budget
        super(NevergradSearch, self).__init__(
            optimizer=optimizer,
            space=space,
            metric=metric,
            mode=mode,
            points_to_evaluate=points_to_evaluate,
            max_concurrent=max_concurrent,
            **kwargs)

    def _setup_nevergrad(self):
        if self._opt_factory:
            self._nevergrad_opt = self._opt_factory(
                parametrization=self._space,
                budget=self._budget,
                num_workers=self.max_concurrent)

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
        if self._parameters is not None and \
           self._nevergrad_opt.dimension != len(self._parameters):
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
    ):

        candidate = self._nevergrad_opt.parametrization.spawn_child(
            new_value=parameters)

        self._nevergrad_opt.tell(candidate, self._metric_op * value)
