import ray
from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import get_dist_info
from mmcv.runner.hooks.logger import LoggerHook
from torch import distributed as dist


@HOOKS.register_module()
class RayTuneLoggerHook(LoggerHook):
    def __init__(
        self,
        interval=1,
        ignore_last=True,
        reset_flag=False,
        by_epoch=False,
        filtering_key='val',
    ):
        super(RayLoggerHook, self).__init__(interval, ignore_last, reset_flag,
                                            by_epoch)
        self.filtering_key = filtering_key

    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        rank, world_size = get_dist_info()
        if world_size > 1:
            if rank == 0:
                broadcasted = [tags]
            else:
                broadcasted = [None]
            dist.broadcast_object_list(broadcasted)
            tags = broadcasted.pop()
        if not dict(
                filter(lambda elem: self.filtering_key in elem[0],
                       tags.items())):
            return
        tags['global_step'] = self.get_iter(runner)
        ray.tune.report(**tags)
