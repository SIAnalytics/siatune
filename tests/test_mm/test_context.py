from unittest.mock import MagicMock

import mmcv
import pytest

from mmtune.mm.context import ContextManager, build_rewriter


def test_contextmanager():
    base_cfg = dict()
    args = MagicMock()

    rewriters = [dict(type='Decouple'), build_rewriter(dict(type='Dump'))]
    context_manager = ContextManager(base_cfg, args, rewriters)

    rewriters = [dict(type='Decouple')]
    context_manager = ContextManager(base_cfg, args, rewriters)
    func = lambda **kargs: 1  # noqa
    inner = context_manager(func)
    config = mmcv.Config(dict())
    context = dict(cfg=config)
    inner(config, context)

    with pytest.raises(TypeError):
        rewriters = [dict(type='Decouple'), []]
        context_manager = ContextManager(base_cfg, args, rewriters)
