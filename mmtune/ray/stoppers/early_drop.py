# Copyright (c) SI-Analytics. All rights reserved.
from collections import defaultdict
from typing import Dict

from ray.tune import Stopper

from .builder import STOPPERS


@STOPPERS.register_module()
class EarlyDroppingStopper(Stopper):
    """Drop Bad Trials."""

    def __init__(self,
                 metric: str,
                 mode: str,
                 metric_threshold: float,
                 grace_period: int = 0) -> None:
        """Initialize early dropping stopper.

        Args:
            metric (str): Metric to optimize.
            mode (str):
                Determines whether objective is
                minimizing or maximizing the metric attribute.
            metric_threshold (float): The threshold for early stopping.
            grace_period (int):
                Only stop trials at least this old in time. Defaults to 0.
        """

        if mode not in ['min', 'max']:
            raise ValueError('mode must be either "min" or "max".')

        self._metric = metric
        self._mode = mode
        self._metric_threshold = metric_threshold
        self._iter = defaultdict(lambda: 0)
        self._grace_period = grace_period

    def __call__(self, trial_id: str, result: Dict) -> bool:
        """Check if trial should be stopped.

        Args:
            trial_id (str): Trial ID.
            result (Dict): Trial result.

        Returns:
            bool: Whether to stop trial.
        """

        self._iter[trial_id] += 1
        metric_result = result.get(self._metric)

        if self._iter[trial_id] < self._grace_period:
            return False

        if self._mode == 'min':
            return True if metric_result > self._metric_threshold else False
        return True if metric_result < self._metric_threshold else False

    def stop_all(self) -> bool:
        """Check if all trials should be stopped.

        Returns:
            bool: Whether to stop all trials.
        """
        return False
