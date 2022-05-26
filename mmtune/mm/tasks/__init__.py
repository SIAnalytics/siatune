from .base import BaseTask
from .blackbox import BlackBoxTask
from .builder import TASKS, build_task_processor
from .mmdet import MMDetection
from .mmseg import MMSegmentation
from .mmtrainbase import MMTrainBasedTask
from .sphere import Sphere

__all__ = [
    'TASKS', 'build_task_processor', 'BaseTask', 'BlackBoxTask',
    'MMTrainBasedTask', 'MMDetection', 'MMSegmentation', 'MMSegmentation',
    'Sphere'
]
