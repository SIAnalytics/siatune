from os import path as osp

from mmtune.utils import get_root_logger


def test_get_root_logger():

    logger = get_root_logger(log_file='test.log')
    logger.info('test')
    assert osp.isfile('test.log')
    assert logger.name == 'mmtune'
