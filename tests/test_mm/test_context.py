import mmcv
import pytest

from mmtune.mm.context import ContextManager


def test_contextmanager():
    rewriters = [dict(type='Decouple', key='cfg')]
    context_manager = ContextManager(rewriters)

    func = lambda **kargs: 1  # noqa
    inner = context_manager(func)
    config = mmcv.Config(dict())
    context = dict(cfg=config)
    inner(**context)

    with pytest.raises(TypeError):
        rewriters = [dict(type='Decouple', key='cfg'), []]
        context_manager = ContextManager(rewriters)
