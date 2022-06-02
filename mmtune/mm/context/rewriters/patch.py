import re
from typing import Dict, Tuple

from .base import BaseRewriter
from .builder import REWRITERS

WRAPPING_REGEXP = r'^\$\((.*)\)$'


def unwrap_regexp(value, regexp=WRAPPING_REGEXP) -> Tuple[str, bool]:
    """Unwrap the value if it is wrapped by the regexp.

    Args:
        value (str): The value to unwrap.
        regexp (str, optional): The regexp to match.
        Defaults to WRAPPING_REGEXP.

    Returns:
        str: The unwrapped value.
        bool: Whether the value is wrapped.
    """
    if not isinstance(value, str):
        return value, False
    searched = re.search(regexp, value)
    if searched:
        value = searched.group(1)
    return value, bool(searched)


@REWRITERS.register_module()
class BatchConfigPathcer(BaseRewriter):
    """Patch the config in the context."""

    def __init__(self, key: str) -> None:
        """Initialize the rewriter.

        Args:
            key (str): The key of the config in the context.
        """
        self.key = key

    def __call__(self, context: Dict) -> Dict:
        """Batch adjustment of elements connected by &.

        Args:
            context (Dict): The context to be rewritten.

        Returns:
            Dict: The context after rewriting.
        """
        cfg = context[self.key]

        for key, value in cfg.items():
            inner_key, is_wrapped = unwrap_regexp(key)
            if not is_wrapped:
                continue
            cfg.pop(key)
            if ' & ' in inner_key:
                for sub_key in inner_key.split(' & '):
                    cfg[sub_key] = value

        return context


@REWRITERS.register_module()
class SequeunceConfigPathcer(BaseRewriter):
    """Patch the config in the context."""

    def __init__(self, key: str) -> None:
        """Initialize the rewriter.

        Args:
            key (str): The key of the config path in the context.
        """

        self.key = key

    def __call__(self, context: Dict) -> Dict:
        """Adjust the elements connected by - in order.

        Args:
            context (Dict): The context to be rewritten.

        Returns:
            Dict: The context after rewriting.
        """
        cfg = context[self.key]

        for key, value in cfg.items():
            inner_key, is_wrapped = unwrap_regexp(key)
            if not is_wrapped:
                continue
            cfg.pop(key)
            if ' - ' in inner_key:
                for idx, sub_key in enumerate(inner_key.split(' - ')):
                    cfg[sub_key] = value[idx]

        return context
