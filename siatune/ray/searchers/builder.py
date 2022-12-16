# Copyright (c) SI-Analytics. All rights reserved.
import inspect

from mmcv.utils import Config, Registry
from ray import tune

SEARCHERS = Registry('searchers')
for func in set(tune.search.SEARCH_ALG_IMPORT.values()):
    if not inspect.isfunction(func):
        continue
    SEARCHERS.register_module(module=func())


def build_searcher(cfg: Config) -> tune.search.Searcher:
    """Build the searcher from configs.

    Args:
        cfg (Config): The configs.

    Returns:
        tune.suggest.Searcher: The searcher.
    """

    return SEARCHERS.build(cfg)
