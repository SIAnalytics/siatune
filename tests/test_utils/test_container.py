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
    container = ImmutableContainer(['too long representation string'])
    assert str(container) == 'IC(*)'
    container = ImmutableContainer({'test': 'test'})
    assert str(container) == "IC({'test': 'test'})"
    assert container.data == {'test': 'test'}
    with pytest.raises(AttributeError):
        container.data = 'modified'
    assert container.alias is None
    assert ImmutableContainer.decouple(container) == {'test': 'test'}

    # test ImmutableContainer w/ alias
    container = ImmutableContainer({'test': 'test'}, 'test')
    assert str(container) == 'test'
    assert container.data == {'test': 'test'}
    with pytest.raises(AttributeError):
        container.data = dict(test='modified')
    assert container.alias == 'test'
    assert ImmutableContainer.decouple(container) == {'test': 'test'}
