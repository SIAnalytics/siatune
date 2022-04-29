from mmcv.utils import Config, Registry
from ray import tune

ALGORITHM = Registry('algorithm')


def build_algorithm(cfg: Config) -> tune.suggest.Searcher:
    return ALGORITHM.build(cfg)
