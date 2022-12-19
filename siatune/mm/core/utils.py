# Copyright (c) SI-Analytics. All rights reserved.

MMENGINE_BASED = False

try:
    from mmengine.config.config import (DELETE_KEY, Config, ConfigDict,
                                        DictAction)
    from mmengine.dist import get_dist_info, master_only
    from mmengine.hooks import CheckpointHook, LoggerHook
    from mmengine.logging import MMLogger as get_logger
    from mmengine.model import is_model_wrapper as is_module_wrapper
    from mmengine.registry import HOOKS, Registry
    from mmengine.runner import Runner as BaseRunner
    from mmengine.runner.checkpoint import get_state_dict, weights_to_cpu
    from mmengine.utils.path import mkdir_or_exist

    MMENGINE_BASED = True

except ImportError:
    try:
        from mmcv.parallel import is_module_wrapper
        from mmcv.runner import HOOKS, BaseRunner
        from mmcv.runner.checkpoint import get_state_dict, weights_to_cpu
        from mmcv.runner.dist_utils import get_dist_info, master_only
        from mmcv.runner.hooks import CheckpointHook
        from mmcv.runner.hooks.logger import LoggerHook
        from mmcv.utils import (DELETE_KEY, Config, ConfigDict, DictAction,
                                Registry, mkdir_or_exist)
        from mmcv.utils.logging import get_logger

    except ImportError:
        raise ImportError(
            'Please install mmengine to use the download command: '
            '`mim install mmengine`.')

__all__ = ['Registry', 'Config', 'get_logger']
