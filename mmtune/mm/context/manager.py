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

        def inner(*searched_cfg, **context):
            context['searched_cfg'] = searched_cfg[0]
            context['base_cfg'] = self.base_cfg
            context['args'] = self.args

            for rewriter in self.rewriters:
                context = rewriter(context)
            return func(**context)

        return inner
