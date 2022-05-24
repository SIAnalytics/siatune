from mmtune.utils.container import ImmutableContainer


def test_immutablecontainer():
    container = ImmutableContainer(dict(model='Model'), 'Model')
