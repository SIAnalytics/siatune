# Copyright (c) SI-Analytics. All rights reserved.
from mmcv.utils import Config, Registry
from ray.tune.search import Searcher

SEARCHERS = Registry('searchers')


def build_searcher(cfg: Config) -> Searcher:
    """Build the searcher from configs.

    Args:
        cfg (Config): The configs.

    Returns:
        tune.suggest.Searcher: The searcher.
    """

    return SEARCHERS.build(cfg)
