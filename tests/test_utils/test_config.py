import tempfile
from os import path as osp

try:
    from mmcv.utils import Config
except ImportError:
    from mmengine.config import Config

from siatune.utils import dump_cfg


def test_dump_cfg():
    cfg = Config(dict())

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = osp.join(tmpdir, 'test.py')
        assert dump_cfg(cfg, save_path)
        assert osp.exists(save_path)
