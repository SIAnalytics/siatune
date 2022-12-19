# Copyright (c) SI-Analytics. All rights reserved.
from mmcv.utils import Config, Registry
from ray.tune.logger import (CSVLoggerCallback, JsonLoggerCallback,
                             LegacyLoggerCallback, LoggerCallback,
                             TBXLoggerCallback)

CALLBACKS = Registry('callbacks')
CALLBACKS.register_module(module=LegacyLoggerCallback)
CALLBACKS.register_module(module=JsonLoggerCallback)
CALLBACKS.register_module(module=CSVLoggerCallback)
CALLBACKS.register_module(module=TBXLoggerCallback)


def build_callback(cfg: Config) -> LoggerCallback:
    return CALLBACKS.build(cfg)
