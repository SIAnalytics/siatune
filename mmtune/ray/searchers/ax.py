from ray.tune.suggest.ax import AxSearch as _AxSearch

from .builder import SEARCHERS


@SEARCHERS.register_module()
class AxSearch(_AxSearch):
    __doc__ = _AxSearch.__doc__
