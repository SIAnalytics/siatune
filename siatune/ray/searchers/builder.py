# Copyright (c) SI-Analytics. All rights reserved.

from mmengine.config import Config
from mmengine.registry import Registry
from ray import tune
from ray.tune.search import Searcher

SEARCHERS = Registry('searchers')
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
