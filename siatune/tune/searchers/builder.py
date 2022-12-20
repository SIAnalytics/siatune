# Copyright (c) SI-Analytics. All rights reserved.

from mmcv.utils import Config, Registry
from ray import tune
from ray.tune.search import Searcher

SEARCHERS = Registry('searchers')

# Dynamically import search_alg
# Refer to https://github.com/ray-project/ray/blob/master/python/ray/tune/search/__init__.py  # noqa
for func in set(tune.search.SEARCH_ALG_IMPORT.values()):
    SEARCHERS.register_module(module=func())


def build_searcher(cfg: Config) -> Searcher:
    """Build the searcher from configs.

    Args:
        cfg (Config): The configs.

    Returns:
        tune.suggest.Searcher: The searcher.
    """

    return SEARCHERS.build(cfg)
