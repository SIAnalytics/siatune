# Copyright (c) SI-Analytics. All rights reserved.
import collections
from typing import List

from .rewriters.builder import build_rewriter


# TODO: Use a context manager as decorator.
class ContextManager:
    """The context manager receives the context from the user and the ray
    tuning algorithm, and refines it into a form usable by the task
    processor."""

    def __init__(self, rewriters: List[dict] = []) -> None:
        """initialize the context manager.

        Args:
            rewriters (List[dict]):
                User-defined context rewriting pipeline.
                Defaults to [].

        Raises:
            TypeError: If the rewriters are not a list.
        """
        self.rewriters = []
        assert isinstance(rewriters, collections.abc.Sequence)
        for rewriter in rewriters:
            if isinstance(rewriter, dict):
                rewriter = build_rewriter(rewriter)
                self.rewriters.append(rewriter)
            elif callable(rewriter):
                self.rewriters.append(rewriter)
            else:
                raise TypeError('rewriter must be callable or a dict')

    def __call__(self, func: callable) -> callable:
        """rewrite the context.

        Args:
            func (callable): The function to be decorated.
        """

        def inner(**context):

            for rewriter in self.rewriters:
                context = rewriter(context)
            return func(**context)

        return inner
