# Copyright (c) SI-Analytics. All rights reserved.

from mmcv.utils import Config, Registry
from ray import tune
from ray.tune.search import Searcher

SEARCHERS = Registry('searchers')
# Remove duplicate :class:`_DummyErrorRaiser`
searchers = set([func() for func in tune.search.SEARCH_ALG_IMPORT.values()])
for cls in searchers:
    SEARCHERS.register_module(module=cls)


def build_searcher(cfg: Config) -> Searcher:
    """Build the searcher from configs.

    Args:
        cfg (Config): The configs.

    Returns:
        tune.suggest.Searcher: The searcher.
    """

    return SEARCHERS.build(cfg)
