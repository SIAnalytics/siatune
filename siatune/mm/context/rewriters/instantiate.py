# Copyright (c) SI-Analytics. All rights reserved.
from typing import Dict, Optional

from mmcv import Config

from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class InstantiateCfg(BaseRewriter):
    """Instantiate the configs in the argparse namespace."""

    def __init__(
        self,
        key: str,
        arg_name: Optional[str] = None,
    ) -> None:
        """Initialize the rewriter.

        Args:
            key (str): The key where the instantiated cfg is stored.
            arg_name (Optional[str]):
                The argparse namespace key where the config path is stored.
        """
        self.key = key
        self.arg_name = arg_name

    def __call__(self, context: Dict) -> Dict:
        """Receive the config path from argparse namespace in the context and
        build the mmcv config file.

        Args:
            context (Dict): The context to be rewritten.

        Returns:
            Dict: The context after rewriting.
        """
        context[self.key] = Config(
            dict()) if self.arg_name is None else Config.fromfile(
                getattr(context.get('args'), self.arg_name))
        return context


@REWRITERS.register_module()
class RawArgInstantiateCfg(BaseRewriter):

    def __init__(self, key: str, arg_idx: int = 0) -> None:
        """Initialize the rewriter.

        Args:
            key (str): The key where the instantiated cfg is stored.
        """
        self.key = key
        self.arg_idx = arg_idx

    def __call__(self, context: Dict) -> Dict:
        """Receive the config path from argparse namespace in the context and
        build the mmcv config file.

        Args:
            context (Dict): The context to be rewritten.

        Returns:
            Dict: The context after rewriting.
        """
        context[self.key] = Config.fromfile(
            context.get('raw_args')[self.arg_idx])
        return context
