from mmcv.utils import Registry

from .base import BaseTask

TASKS = Registry('tasks')


def build_task_processor(task_name: str) -> BaseTask:
    return TASKS.build(dict(type=task_name))
