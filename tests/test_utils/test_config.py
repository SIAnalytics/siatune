import tempfile
from os import path as osp

from mmcv.utils import Config

from mmtune.utils import dump_cfg


def test_dump_cfg():
    cfg = Config(dict())

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = osp.join(tmpdir, 'test.py')
        assert dump_cfg(cfg, save_path)
        assert osp.exists(save_path)
