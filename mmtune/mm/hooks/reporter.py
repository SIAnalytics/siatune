import ray
from mmcv.runner import HOOKS, BaseRunner
from mmcv.runner.dist_utils import get_dist_info
from mmcv.runner.hooks.logger import LoggerHook
from torch import distributed as dist


@HOOKS.register_module()
class RayTuneLoggerHook(LoggerHook):
    """MMCV Logger hook for Ray Tune."""

    def __init__(
            self,
            interval: int = 1,
            ignore_last: bool = True,
            reset_flag: bool = False,
            by_epoch: bool = False,
            filtering_key: str = 'val',
    ) -> None:
        """Initialize the hook.

        Args:
            interval (int): The interval to log.
            ignore_last (bool): Whether to ignore the last iteration.
            reset_flag (bool): Whether to reset the iteration.
            by_epoch (bool): Whether to log by epoch.
            filtering_key (str): The key to filter.
        """
        super(RayTuneLoggerHook, self).__init__(interval, ignore_last,
                                                reset_flag, by_epoch)
        self.filtering_key = filtering_key

    def log(self, runner: BaseRunner) -> None:
        """Log the information.

        Args:
            runner (:obj:`mmcv.runner.BaseRunner`): The runner to log.
        """

        tags = self.get_loggable_tags(runner)
        rank, world_size = get_dist_info()
        if world_size > 1:
            if rank == 0:
                broadcasted = [tags]
            else:
                broadcasted = [None]
            dist.broadcast_object_list(broadcasted)
            tags = broadcasted.pop()
        if not any(
                filter(lambda elem: self.filtering_key in elem, tags.keys())):
            return
        tags['global_step'] = self.get_iter(runner)
        ray.tune.report(**tags)
