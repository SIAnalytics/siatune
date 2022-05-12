import argparse
import collections
from typing import List, Optional

from mmcv.utils import Config

from .rewriters.builder import build_rewriter


class ContextManager:

    def __init__(self,
                 base_cfg: Optional[Config] = None,
                 args: Optional[argparse.Namespace] = None,
                 rewriters: List[dict] = []):
        self.base_cfg = base_cfg
        self.args = args
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

        def inner(*tpargs, **kwargs):
            kwargs['searched_cfg'] = tpargs[0]
            kwargs['base_cfg'] = self.base_cfg
            kwargs['args'] = self.args

            for rewriter in self.rewriters:
                kwargs = rewriter(kwargs)
            return func(**kwargs)

        return inner
