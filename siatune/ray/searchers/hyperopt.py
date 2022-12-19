# Copyright (c) SI-Analytics. All rights reserved.
from ray.tune.search.hyperopt import HyperOptSearch as _HyperOptSearch

from .builder import SEARCHERS


@SEARCHERS.register_module()
class HyperOptSearch(_HyperOptSearch):
    __doc__ = _HyperOptSearch.__doc__
