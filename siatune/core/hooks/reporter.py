# Copyright (c) SI-Analytics. All rights reserved.
from mmcv.runner import HOOKS, BaseRunner
from mmcv.runner.dist_utils import get_dist_info, master_only
from mmcv.runner.hooks.logger import LoggerHook
from ray.air import session
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

    def after_train_iter(self, runner: BaseRunner) -> None:
        """Log after train itr.

        Args:
            runner (:obj:`mmcv.runner.BaseRunner`): The runner to log.
        """
        if self.by_epoch and self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif not self.by_epoch and self.every_n_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            runner.log_buffer.average(self.interval)

        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()

    def after_train_epoch(self, runner: BaseRunner) -> None:
        """Log after train epoch.

        Args:
            runner (:obj:`mmcv.runner.BaseRunner`): The runner to log.
        """
        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()

    def after_val_epoch(self, runner: BaseRunner) -> None:
        """Log after val epoch.

        Args:
            runner (:obj:`mmcv.runner.BaseRunner`): The runner to log.
        """
        runner.log_buffer.average()
        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()

    @master_only
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
        session.report(tags)
