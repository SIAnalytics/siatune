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
        self._data = data
        self._alias = alias
        super().__init__()

    def __repr__(self) -> str:
        """Return the string representation of the container."""
        if self._alias is not None:
            return self._alias
        elif len(self._data.__repr__()) > ImmutableContainer.MAX_REPR_LEN:
            return f'{self.__class__.__name__}( * )'
        return f'{self.__class__.__name__}( {repr(self.data)} )'

    @property
    def data(self):
        """Return the data."""
        return self._data

    @property
    def alias(self):
        """Return the alias."""
        return self._alias

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
