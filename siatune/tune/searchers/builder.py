# Copyright (c) SI-Analytics. All rights reserved.

from mmengine.config import Config
from mmengine.registry import Registry
from ray import tune
from ray.tune.search import Searcher

SEARCHERS = Registry('searcher')

# Dynamically import search_alg
# Refer to https://github.com/ray-project/ray/blob/master/python/ray/tune/search/__init__.py  # noqa
for cls in set(func() for func in tune.search.SEARCH_ALG_IMPORT.values()):
    SEARCHERS.register_module(module=cls)


def build_searcher(cfg: Config) -> Searcher:
    """Build the searcher from configs.

    Args:
        cfg (Config): The configs.

    Returns:
        tune.suggest.Searcher: The searcher.
    """

    return SEARCHERS.build(cfg)
