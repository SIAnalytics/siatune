import pytest

from mmtune.utils.container import ImmutableContainer, _Freezer


def test_freezer():
    freezer = _Freezer()
    with pytest.raises(AttributeError):
        freezer._lock = False
    with pytest.raises(AttributeError):
        del freezer._lock


def test_immutablecontainer():
    container = ImmutableContainer(dict(test='test'), 'test')
    assert str(container) == 'test'
    assert str(container.data) == dict(test='test')
    with pytest.raises(AttributeError):
        container.data = dict(test='modified')
    assert container.alias == 'test'
    assert container == ImmutableContainer(dict(test='test'), 'new test')
    assert ImmutableContainer.decouple(container) == dict(test='test')
