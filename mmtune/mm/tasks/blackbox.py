from typing import Callable

from .base import BaseTask
from .builder import TASK


@TASK.register_module()
class BloackBoxTask(BaseTask):

    @staticmethod
    def create_trainable() -> Callable:
        return BloackBoxTask.run
