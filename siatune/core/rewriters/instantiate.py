# Copyright (c) SI-Analytics. All rights reserved.
import argparse
from typing import Dict

from mmengine.config import Config

from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class InstantiateCfg(BaseRewriter):
    """Instantiate the configs in the argparse namespace."""
    arg_name: str = 'config'
    raw_arg_idx: int = 0

    def __init__(
        self,
        key: str,
    ) -> None:
        """Initialize the rewriter.

        Args:
            key (str): The key where the instantiated cfg is stored.
        """
        self.key = key

    def __call__(self, context: Dict) -> Dict:
        """Receive the config path from argparse namespace in the context and
        build the mmcv config file.

        Args:
            context (Dict): The context to be rewritten.

        Returns:
            Dict: The context after rewriting.
        """
        is_parsed = isinstance(context['args'], argparse.Namespace)
        file_name: str
        if is_parsed:
            file_name = getattr(context['args'], self.arg_name)
        else:
            file_name = context['args'][self.raw_arg_idx]
        context[self.key] = Config.fromfile(file_name)
        return context
