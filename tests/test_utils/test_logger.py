import logging

from siatune.utils import get_root_logger


def test_get_root_logger():
    logger = get_root_logger()
    logger.info('test')
    assert isinstance(logger, logging.Logger)
    assert logger.name == 'siatune'
