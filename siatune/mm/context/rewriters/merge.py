# Copyright (c) SI-Analytics. All rights reserved.
from mmengine.config.config import DELETE_KEY, Config, ConfigDict

from .base import BaseRewriter
from .builder import REWRITERS


@REWRITERS.register_module()
class MergeConfig(BaseRewriter):
    """Merge the two configs."""

    def __init__(self, src_key: str, dst_key: str, key: str):
        """Initialize the MergeConfig class.

        Args:
            src_key (str): The key of the configs.
            dst_key (str): The key of the configs.
            key (str):
                The context key where the merged config will be stored.
        """
        self.src_key = src_key
        self.dst_key = dst_key
        self.key = key

    @staticmethod
    def merge_dict(src: dict, dst: dict, allow_list_keys: bool = False):
        """merge dict ``a`` into dict ``b`` (non-inplace).
        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.
        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in source ``a`` and will replace the element of the
              corresponding index in b if b is a list. Default: False.
        Returns:
            dict: The modified dict of ``b`` using ``a``.
        Examples:
            # Normally merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}
            # Delete b first and merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}
            # b is a list
            >>> Config._merge_a_into_b(
            ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)
            [{'a': 2}, {'b': 2}]

        """
        dst = dst.copy()
        for k, v in src.items():
            if allow_list_keys and k.isdigit() and isinstance(dst, list):
                k = int(k)
                if len(dst) <= k:
                    raise KeyError(
                        f'Index {k} exceeds the length of list {dst}')
                # modified from the mmcv.config.Config._merge_a_into_b
                # this allows merging with primitives such as int, float
                dst[k] = MergeConfig.merge_dict(v, dst[k],
                                                allow_list_keys) if hasattr(
                                                    dst[k], 'copy') else v
            elif isinstance(v, dict):
                if k in dst and not v.pop(DELETE_KEY, False):
                    allowed_types = (dict, list) if allow_list_keys else dict
                    if not isinstance(dst[k], allowed_types):
                        raise TypeError(
                            f'{k}={v} in child config cannot inherit from '
                            f'base because {k} is a dict in the child config '
                            f'but is of type {type(dst[k])} in base config. '
                            f'You may set `{DELETE_KEY}=True` to ignore the '
                            f'base config.')
                    dst[k] = MergeConfig.merge_dict(v, dst[k], allow_list_keys)
                else:
                    dst[k] = ConfigDict(v)
            else:
                dst[k] = v
        return dst

    def __call__(self, context: dict, allow_list_keys: bool = True) -> dict:
        """Semantically merge and save the two configs.

        Args:
            context (dict): The context to be rewritten.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
                are allowed in source ``a`` and will replace the element of the
                corresponding index in b if b is a list. Default: True.

        Returns:
            dict: The context after rewriting.
        """
        src = context.pop(self.src_key)
        dst = context.pop(self.dst_key)
        unpacked_src = {}
        for full_key, v in src.items():
            d = unpacked_src
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v
        context[self.key] = Config(
            self.merge_dict(
                unpacked_src,
                dst.__getattribute__('_cfg_dict'),
                allow_list_keys=allow_list_keys),
            cfg_text=dst.text,
            filename=dst.filename)
        return context
