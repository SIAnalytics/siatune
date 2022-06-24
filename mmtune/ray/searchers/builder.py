from mmcv.utils import Config, Registry
from ray import tune

SEARCHERS = Registry('searchers')


def build_searcher(cfg: Config) -> tune.suggest.Searcher:
    """Build the searcher from configs.

    Args:
        cfg (Config): The configs.

    Returns:
        tune.suggest.Searcher: The searcher.
    """

    return SEARCHERS.build(cfg)
