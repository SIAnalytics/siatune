import tempfile
from os import path as osp

from mmtune.utils import get_root_logger


def test_get_root_logger():

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = osp.join(tmpdir, 'test.log')
        logger = get_root_logger(log_file=save_path)
        logger.info('test')
        assert osp.exists(save_path)
        assert logger.name == 'mmtune'
