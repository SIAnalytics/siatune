from typing import List

from .builder import REWRITERS


@REWRITERS.register_module()
class CustomHookRegister:

    def __init__(self, key: str, post_custom_hooks: List[str]) -> None:
        self.post_custom_hooks = post_custom_hooks
        self.key = key

    def __call__(self, context: dict) -> dict:
        custom_hooks = getattr(context['cfg'], 'custom_hooks', []).copy()
        for custom_hook in self.post_custom_hooks:
            custom_hooks.append(custom_hook)
        context[self.key].custom_hooks = custom_hooks
        return context
