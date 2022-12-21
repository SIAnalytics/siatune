# Copyright (c) SI-Analytics. All rights reserved.
from .base import BaseTask
from .blackbox import BlackBoxTask
from .builder import TASKS, build_task_processor
from .cont_test_func import ContinuousTestFunction
from .disc_test_func import DiscreteTestFunction
from .mmcls import MMClassification
from .mmdet import MMDetection
from .mmedit import MMEditing
from .mmseg import MMSegmentation
from .mm import MMBaseTask

__all__ = [
    'TASKS',
    'build_task_processor',
    'BaseTask',
    'BlackBoxTask',
    'ContinuousTestFunction',
    'DiscreteTestFunction',
    'MMBaseTask',
    'MMClassification',
    'MMDetection',
    'MMEditing',
    'MMSegmentation',
]
