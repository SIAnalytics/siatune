from mmcv.utils import Config, Registry

REWRITERS = Registry('rewriters')


def build_rewriter(cfg: Config) -> object:
    return REWRITERS.build(cfg)
