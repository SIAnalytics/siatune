import pytest

from mmtune.ray.stoppers import EarlyDroppingStopper


def test_earlydroppingstopper():
    with pytest.raises(ValueError):
        EarlyDroppingStopper(metric='acc', mode='more', metric_threshold=0.5)

    stopper = EarlyDroppingStopper(
        metric='acc', mode='max', metric_threshold=0.5)
    print(stopper('abc', dict(acc=0.6)))
