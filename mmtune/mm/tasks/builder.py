from mmcv.utils import Registry

TASKS = Registry('tasks')


def build_task_processor(task_name: str):
    return TASKS.build(dict(type=task_name))
