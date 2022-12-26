# Copyright (c) SI-Analytics. All rights reserved.
from typing import Dict

from mmengine.registry import Registry

from .base import BaseTask

TASKS = Registry('task')


def build_task(task: Dict) -> BaseTask:
    """Build the task processor.

    Args:
        task (Dict): The task config to build.

    Returns:
        BaseTask: The task processor.
    """

    return TASKS.build(task)
