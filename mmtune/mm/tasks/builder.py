from mmcv.utils import Registry

TASKS = Registry('tasks')


def build_task_processor(task_name):
    TASKS.build(dict(type=task_name))
