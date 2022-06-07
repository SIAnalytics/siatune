import pytest

from mmtune.mm.context import ContextManager


def test_contextmanager():
    with pytest.raises(TypeError):
        ContextManager(['test'])

    class Rewriter:

        def __call__(self, context):
            return dict(test='test')

    context_manager = ContextManager([Rewriter()])
    assert context_manager(lambda **context: context)(test='fake') == dict(
        test='test')
