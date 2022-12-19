# Copyright (c) SI-Analytics. All rights reserved.
from ray.air import session
from torch import distributed as dist

from siatune.mm.core import (HOOKS, MMENGINE_BASED, BaseRunner, LoggerHook,
                             get_dist_info)


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
        if MMENGINE_BASED:
            kwargs = dict(
                interval=interval,
                ignore_last=ignore_last,
                log_metric_by_epoch=by_epoch)
        else:
            kwargs = dict(
                interval=interval,
                ignore_last=ignore_last,
                reset_flag=reset_flag,
                by_epoch=by_epoch)

        super(RayTuneLoggerHook, self).__init__(**kwargs)
        self.filtering_key = filtering_key

    def after_train_iter(self, runner: BaseRunner, **kwargs) -> None:
        """Log after train itr.

        Args:
            runner (:obj:`mmcv.runner.BaseRunner`): The runner to log.
        """
        if MMENGINE_BASED:
            batch_idx = kwargs['batch_idx']
            if self.every_n_train_iters(
                    runner, self.interval_exp_name) or (self.end_of_epoch(
                        runner.train_dataloader, batch_idx)):
                exp_info = f'Exp name: {runner.experiment_name}'
                runner.logger.info(exp_info)
            if self.every_n_inner_iters(batch_idx, self.interval):
                tag, log_str = runner.log_processor.get_log_after_iter(
                    runner, batch_idx, 'train')
            elif (self.end_of_epoch(runner.train_dataloader, batch_idx)
                  and not self.ignore_last):
                # `runner.max_iters` may not be divisible by `self.interval`. if
                # `self.ignore_last==True`, the log of remaining iterations will
                # be recorded (Epoch [4][1000/1007], the logs of 998-1007
                # iterations will be recorded).
                tag, log_str = runner.log_processor.get_log_after_iter(
                    runner, batch_idx, 'train')
            else:
                return
            runner.logger.info(log_str)

            # TODO: Here we sohuld feed tags to ray reporter

            raise RuntimeError

            return super().after_train_iter(runner, **kwargs)

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
