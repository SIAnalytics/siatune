import re

from .builder import REWRITERS

WRAPPING_REGEXP = r'^\$\((.*)\)$'


def unwrap_regexp(value, regexp=WRAPPING_REGEXP):
    if not isinstance(value, str):
        return value, False
    searched = re.search(regexp, value)
    if searched:
        value = searched.group(1)
    return value, bool(searched)


@REWRITERS.register_module()
class BatchConfigPathcer:

    def __call__(self, context: dict):
        cfg = context['searched_cfg']

        for key, value in cfg.items():
            inner_key, is_wrapped = unwrap_regexp(key)
            if not is_wrapped:
                continue
            cfg.pop(key)
            if ' & ' in inner_key:
                for sub_key in inner_key.split(' & '):
                    cfg[sub_key] = value

        return context


@REWRITERS.register_module()
class SequeunceConfigPathcer:

    def __call__(self, context: dict):
        cfg = context['searched_cfg']

        for key, value in cfg.items():
            inner_key, is_wrapped = unwrap_regexp(key)
            if not is_wrapped:
                continue
            cfg.pop(key)
            if ' - ' in inner_key:
                for idx, sub_key in enumerate(inner_key.split(' - ')):
                    cfg[sub_key] = value[idx]

        return context
