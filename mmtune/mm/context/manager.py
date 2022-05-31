import collections
from typing import List

from .rewriters.builder import build_rewriter


class ContextManager:

    def __init__(self, rewriters: List[dict] = []):
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

    def __call__(self, func):

        def inner(**context):

            for rewriter in self.rewriters:
                context = rewriter(context)
            return func(**context)

        return inner
