from .base import BaseTask
from .blackbox import BloackBoxTask
from .mmseg import MMSegmentation
from .mmtrainbase import MMTrainBasedTask
from .sphere import Sphere

__all__ = [
    'BaseTask', 'BloackBoxTask', 'MMTrainBasedTask', 'MMSegmentation',
    'MMSegmentation', 'Sphere'
]
