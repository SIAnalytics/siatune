import pytest

from siatune.utils.container import ImmutableContainer, _Freezer


def test_freezer():
    freezer = _Freezer()
    with pytest.raises(AttributeError):
        freezer._lock = False
    with pytest.raises(AttributeError):
        del freezer._lock


def test_container():
    # test ImmutableContainer w/o alias
    container = ImmutableContainer(dict(test='test'))
    assert str(container) == "ImmutableContainer({'test': 'test'})"
    assert container.data == dict(test='test')
    with pytest.raises(AttributeError):
        container.data = dict(test='modified')
    assert container.alias is None
    assert ImmutableContainer.decouple(container) == dict(test='test')

    # test ImmutableContainer w/ alias
    container = ImmutableContainer(dict(test='test'), 'test')
    assert str(container) == 'test'
    assert container.data == dict(test='test')
    with pytest.raises(AttributeError):
        container.data = dict(test='modified')
    assert container.alias == 'test'
    assert ImmutableContainer.decouple(container) == dict(test='test')
