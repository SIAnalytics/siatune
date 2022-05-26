from mmcv.utils import Registry

TASKS = Registry('tasks')


def build_task_processor(task: dict):
    return TASKS.build(task)
