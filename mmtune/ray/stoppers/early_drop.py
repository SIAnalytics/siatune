from collections import defaultdict

from ray.tune import Stopper

from .builder import STOPPER


@STOPPER.register_module()
class EarlyDroppingStopper(Stopper):

    def __init__(self,
                 metric: str,
                 mode: str,
                 metric_threshold: float,
                 grace_period: int = 0):

        assert mode in ['min', 'max']
        self._metric = metric
        self._mode = mode
        self._metric_threshold = metric_threshold
        self._iter = defaultdict(lambda: 0)
        self._grace_period = grace_period

    def __call__(self, trial_id: str, result: dict) -> bool:
        self._iter[trial_id] += 1
        metric_result = result.get(self._metric)

        if self._iter[trial_id] < self._grace_period:
            return False

        if self._mode == 'min':
            return True if metric_result > self._metric_threshold else False
        return True if metric_result < self._metric_threshold else False

    def stop_all(self) -> bool:
        return False
