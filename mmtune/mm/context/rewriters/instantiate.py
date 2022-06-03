from typing import Dict, Optional

from mmcv import Config

from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class InstantiateCfg(BaseRewriter):
    """Instantiate the configs in the argparse namespace."""

    def __init__(
            self,
            dst_key: str,
            arg_key: Optional[str] = None,
    ) -> None:
        """Initialize the rewriter.

        Args:
            dst_key (str): The key where the instantiated cfg is stored.
            arg_key (Optional[str]):
                The argparse namespace key where the config path is stored.
        """
        self.dst_key = dst_key
        self.arg_key = arg_key

    def __call__(self, context: Dict) -> Dict:
        """Receive the config path from argparse namespace in the context and
        build the mmcv config file.

        Args:
            context (Dict): The context to be rewritten.

        Returns:
            Dict: The context after rewriting.
        """
        context[self.dst_key] = Config(
            dict()) if self.arg_key is None else Config.fromfile(
                getattr(context.get('args'), self.arg_key))
        return context
