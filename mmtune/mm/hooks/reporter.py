from typing import Optional

import ray
from mmcv.runner import HOOKS, BaseRunner, master_only
from mmcv.runner.hooks.logger import LoggerHook


@HOOKS.register_module()
class RayTuneLoggerHook(LoggerHook):
    """MMCV Logger hook for Ray Tune."""

    def __init__(
            self,
            interval: int = 1,
            ignore_last: bool = True,
            reset_flag: bool = False,
            by_epoch: bool = False,
            filter_key: Optional[str] = None,
    ) -> None:
        """Initialize the hook.

        Args:
            interval (int): The interval to log.
            ignore_last (bool): Whether to ignore the last iteration.
            reset_flag (bool): Whether to reset the iteration.
            by_epoch (bool): Whether to log by epoch.
            filter_key (str, optional): The key to filter. Default: None.
        """
        super(RayTuneLoggerHook, self).__init__(interval, ignore_last,
                                                reset_flag, by_epoch)
        self.filter_key = filter_key

    @master_only
    def log(self, runner: BaseRunner) -> None:
        """Log the information.

        Args:
            runner (:obj:`mmcv.runner.BaseRunner`): The runner to log.
        """

        tags = self.get_loggable_tags(runner)
        if self.filter_key is not None:
            tags = dict(
                filter(lambda key, _: key.startswith(self.filter_key),
                       tags.items()))
        tags['global_step'] = self.get_iter(runner)
        ray.tune.report(**tags)
