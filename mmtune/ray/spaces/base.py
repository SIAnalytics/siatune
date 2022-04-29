from abc import ABCMeta


class BaseSpace(metaclass=ABCMeta):

    @property
    def space(self):
        return getattr(self, '_space', None)
