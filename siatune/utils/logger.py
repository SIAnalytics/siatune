# Copyright (c) SI-Analytics. All rights reserved.
import logging
from typing import Optional

from mmengine.logging import MMLogger


def get_root_logger(log_file: Optional[str] = None,
                    log_level: int = logging.INFO):
    """Get the root logger. The logger will be initialized if it has not been
    initialized. By default a StreamHandler will be added. If `log_file` is
    specified, a FileHandler will also be added. The name of the root logger is
    the top-level package name, e.g., "siatune".

    Args:
        log_file (Optional[str]): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
    Returns:
        logging.Logger: The root logger.
    """

    logger = MMLogger(name='SIATune', log_file=log_file, log_level=log_level)

    return logger
