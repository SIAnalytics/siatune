# Copyright (c) SI-Analytics. All rights reserved.
from .base import BaseTask
from .blackbox import BlackBoxTask
from .builder import TASKS, build_task_processor
from .bystandertrain import BystanderTrainBasedTask
from .cont_test_func import ContinuousTestFunction
from .disc_test_func import DiscreteTestFunction
from .mmcls import MMClassification
from .mmdet import MMDetection
from .mmseg import MMSegmentation
from .mmtrainbase import MMTrainBasedTask

__all__ = [
    'MMClassification',
    'DiscreteTestFunction',
    'ContinuousTestFunction',
    'TASKS',
    'build_task_processor',
    'BaseTask',
    'BlackBoxTask',
    'MMTrainBasedTask',
    'MMDetection',
    'MMSegmentation',
    'MMSegmentation',
    'BystanderTrainBasedTask',
]
