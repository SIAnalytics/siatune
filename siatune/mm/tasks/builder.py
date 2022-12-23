# Copyright (c) SI-Analytics. All rights reserved.
from typing import Dict

from mmengine.registry import Registry

from .base import BaseTask

TASKS = Registry('tasks')


def build_task_processor(task: Dict) -> BaseTask:
    """Build the task processor.

    Args:
        task (Dict): The task config to build.

    Returns:
        BaseTask: The task processor.
    """

    return TASKS.build(task)
