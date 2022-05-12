from .base import BaseTask
from .blackbox import BloackBoxTask
from .builder import TASKS, build_task_processor
from .mmseg import MMSegmentation
from .mmtrainbase import MMTrainBasedTask
from .sphere import Sphere

__all__ = [
    'TASKS', 'build_task_processor', 'BaseTask', 'BloackBoxTask',
    'MMTrainBasedTask', 'MMSegmentation', 'MMSegmentation', 'Sphere'
]
