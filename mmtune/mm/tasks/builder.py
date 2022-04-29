from mmcv.utils import Registry

TASK = Registry('task')


def build_task_processor(task_name):
    TASK.build(dict(type=task_name))
