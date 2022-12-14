# Copyright (c) SI-Analytics. All rights reserved.
import copy
from typing import Any, Optional


class _Freezer(object):
    """Freeze any class, such that instantiated objects become immutable.

    https://medium.datadriveninvestor.com/immutability-in-python-d57a3b23f336
    """

    _lock = False

    def __init__(self):
        """Initialize the freezer."""
        self._lock = True

    def __delattr__(self, *args, **kwargs) -> None:
        """Delete an attribute.

        Raises:
            AttributeError: If the object is locked.
        """

        if self._lock:
            raise AttributeError
        object.__delattr__(self, *args, **kwargs)

    def __setattr__(self, *args, **kwargs) -> None:
        """Set an attribute.

        Raises:
            AttributeError: If the object is locked.
        """

        if self._lock:
            raise AttributeError
        object.__setattr__(self, *args, **kwargs)


class ImmutableContainer(_Freezer):
    """Ensure object immutability."""

    MAX_REPR_LEN: int = 16

    def __init__(self, data: Any, alias: Optional[str] = None) -> None:
        """Initialize the container.

        Args:
            data (Any): The data to be stored.
            alias (Optional[str]): The alias of the data.
        """
        self.__data = data
        self._alias = alias
        super().__init__()

    def __repr__(self) -> str:
        """Return the string representation of the container."""
        if self._alias is not None:
            return self._alias
        elif len(self.__data.__repr__()) > ImmutableContainer.MAX_REPR_LEN:
            return f'{self.__class__.__name__}( * )'
        return f'{self.__class__.__name__}( {repr(self.data)} )'

    @property
    def data(self):
        """Return the data."""
        return copy.deepcopy(self.__data)

    @data.setter
    def data(self):
        raise AttributeError(
            'Setting data inside an immutable container is not allowed.')

    @property
    def alias(self):
        """Return the alias."""
        return self._alias

    def __eq__(self, other: Any) -> bool:
        """Return True if the containers are equal.

        Args:
            other (Any): The other object to compare.
        Returns:
            bool: True if the other object is equal.
        """
        if isinstance(other, ImmutableContainer):
            return self.data == other.data
        return self.data == other

    def __hash__(self) -> int:
        """Return hash value of `str(self.data)`. It needs to use FLAML.

        Returns:
            int: hash value.
        """
        return hash(str(self.data))

    @classmethod
    def decouple(cls, inputs: Any) -> Any:
        """Decouple the inputs.

        Args:
            inputs (Any): The inputs to be decoupled.

        Returns:
            Any: The decoupled inputs.
        """
        if isinstance(inputs, ImmutableContainer):
            return inputs.data
        elif isinstance(inputs, dict):
            outputs = inputs.copy()
            for k, v in inputs.items():
                outputs[k] = ImmutableContainer.decouple(v)
            return outputs
        elif isinstance(inputs, list):
            outputs = inputs.copy()
            for idx, elm in enumerate(inputs):
                outputs[idx] = ImmutableContainer.decouple(elm)
            return outputs
        return inputs
