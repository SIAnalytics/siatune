# Copyright (c) SI-Analytics. All rights reserved.
from mmengine.config import Config
from mmengine.registry import Registry
from ray.tune.logger import (CSVLoggerCallback, JsonLoggerCallback,
                             LegacyLoggerCallback, LoggerCallback,
                             TBXLoggerCallback)

CALLBACKS = Registry('callback')
CALLBACKS.register_module(module=LegacyLoggerCallback)
CALLBACKS.register_module(module=JsonLoggerCallback)
CALLBACKS.register_module(module=CSVLoggerCallback)
CALLBACKS.register_module(module=TBXLoggerCallback)


def build_callback(cfg: Config) -> LoggerCallback:
    return CALLBACKS.build(cfg)
