from abc import ABCMeta


class BaseSpace(metaclass=ABCMeta):
    """Base Space class."""

    @property
    def space(self) -> callable:
        """Return the space."""
        return getattr(self, '_space', None)
