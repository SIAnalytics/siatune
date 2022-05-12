from mmcv.utils import Config, Registry
from ray import tune

SEARCHERS = Registry('searchers')


def build_searcher(cfg: Config) -> tune.suggest.Searcher:
    return SEARCHERS.build(cfg)
