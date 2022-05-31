from typing import Dict, Optional

from mmcv import Config

from .builder import REWRITERS


@REWRITERS.register_module()
class BuildBaseCfg:

    def __init__(
        self,
        dst_key: str,
        arg_key: Optional[str] = None,
    ):
        self.dst_key = dst_key
        self.arg_key = arg_key

    def __call__(self, context: Dict):
        context[self.dst_key] = Config(
            dict()) if self.arg_key is None else Config.fromfile(
                getattr(context.get('args'), self.arg_key))
        return context
