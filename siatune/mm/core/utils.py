# Copyright (c) SI-Analytics. All rights reserved.

MMENGINE_BASED = False

try:
    from mmengine.registry import Registry, HOOKS
    from mmengine.config.config import Config, ConfigDict, DELETE_KEY, DictAction
    from mmengine.logging import MMLogger as get_logger
    from mmengine.model import is_model_wrapper as is_module_wrapper
    from mmengine.runner import Runner as BaseRunner
    from mmengine.runner.checkpoint import get_state_dict, weights_to_cpu
    from mmengine.dist import master_only, get_dist_info
    from mmengine.hooks import LoggerHook, CheckpointHook
    from mmengine.utils.path import mkdir_or_exist
   
    MMENGINE_BASED = True

except ImportError:
    try:
        from mmcv.utils import Registry, Config, ConfigDict, DELETE_KEY, DictAction
        from mmcv.utils.logging import get_logger
        from mmcv.parallel import is_module_wrapper
        from mmcv.runner import HOOKS, BaseRunner
        from mmcv.runner.checkpoint import get_state_dict, weights_to_cpu
        from mmcv.runner.dist_utils import master_only, get_dist_info
        from mmcv.runner.hooks import CheckpointHook
        from mmcv.runner.hooks.logger import LoggerHook
        from mmcv.utils import mkdir_or_exist


    except ImportError:
        raise ImportError(
            'Please install mmengine to use the download command: '
                '`mim install mmengine`.')

__all__ = ['Registry', 'Config', 'get_logger']