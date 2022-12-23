# Copyright (c) SI-Analytics. All rights reserved.
from mmengine.config import Config


def dump_cfg(cfg: Config, save_path: str) -> bool:
    """Dump the config. It has been split into a separate function to support
    the lower version of mmcv.

    Args:
        cfg (Config): The config to be dumped.
        save_path (str): The path to save the config.

    Returns:
        bool: Whether the config is dumped successfully.
    """
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(cfg.pretty_text)
        return True
    except Exception as err:
        from siatune.utils import get_root_logger
        logger = get_root_logger()
        logger.error(f'Failed to dump config to {save_path}: {err}')
        return False
