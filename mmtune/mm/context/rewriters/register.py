from typing import List

from .builder import REWRITERS


@REWRITERS.register_module()
class CustomHookRegister:

    def __init__(self,
                 ctx_cfg: str,
                 post_custom_hooks: List[str],
                 hk_key: str = 'custom_hooks') -> None:
        self.post_custom_hooks = post_custom_hooks
        self.ctx_cfg = ctx_cfg
        self.hk_key = hk_key

    def __call__(self, context: dict) -> dict:
        custom_hooks = getattr(context[self.ctx_cfg], self.hk_key, []).copy()
        for custom_hook in self.post_custom_hooks:
            custom_hooks.append(custom_hook)
        setattr(context[self.ctx_cfg], self.hk_key, custom_hooks)
        return context
