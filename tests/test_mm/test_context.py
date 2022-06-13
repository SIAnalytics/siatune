import pytest

from mmtune.mm.context import REWRITERS, ContextManager


def test_contextmanager():
    with pytest.raises(TypeError):
        ContextManager(['test'])

    @REWRITERS.register_module()
    class TestRewriter:

        def __call__(self, context):
            return dict(test='test')

    context_manager = ContextManager([TestRewriter()])
    assert context_manager(lambda **context: context)(test='fake') == dict(
        test='test')

    dict_init_context_manager = ContextManager([dict(type='TestRewriter')])
    assert dict_init_context_manager(lambda **context: context)(
        test='fake') == dict(test='test')
