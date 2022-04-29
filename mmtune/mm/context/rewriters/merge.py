from typing import Union

from mmcv.utils import Config, ConfigDict
from mmcv.utils.config import DELETE_KEY

from .builder import REWRITER


@REWRITER.register_module()
class ConfigMerger:

    @staticmethod
    def merge_dict(src: dict,
                   dst: dict,
                   allow_list_keys: Union[list, dict, bool] = False):
        dst = dst.copy()
        for k, v in src.items():
            if allow_list_keys and k.isdigit() and isinstance(dst, list):
                k = int(k)
                if len(dst) <= k:
                    raise KeyError(
                        f'Index {k} exceeds the length of list {dst}')
                dst[k] = ConfigMerger.merge_dict(v, dst[k],
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
                    dst[k] = ConfigMerger.merge_dict(v, dst[k],
                                                     allow_list_keys)
                else:
                    dst[k] = ConfigDict(v)
            else:
                dst[k] = v
        return dst

    def __call__(self, context: dict, allow_list_keys=True):
        src = context.pop('searched_cfg')
        dst = context.pop('base_cfg')
        unpacked_src = {}
        for full_key, v in src.items():
            d = unpacked_src
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v
        context['cfg'] = Config(
            self.merge_dict(
                unpacked_src,
                dst.__getattribute__('_cfg_dict'),
                allow_list_keys=allow_list_keys),
            cfg_text=dst.text,
            filename=dst.filename)

        return context
