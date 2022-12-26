# Copyright (c) SI-Analytics. All rights reserved.
from pathlib import Path
from typing import Dict, Optional, Union

from mmengine.dist import get_dist_info
from mmengine.hooks import LoggerHook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from ray.air import session
from torch import distributed as dist

from siatune.version import IS_DEPRECATED_MMCV

if not IS_DEPRECATED_MMCV:
    from mmengine.hooks.logger_hook import SUFFIX_TYPE

    @HOOKS.register_module()
    class RayTuneLoggerHook(LoggerHook):
        """MMCV Logger hook for Ray Tune."""

        def __init__(self,
                     interval: int = 10,
                     ignore_last: bool = True,
                     interval_exp_name: int = 1000,
                     out_dir: Optional[Union[str, Path]] = None,
                     out_suffix: SUFFIX_TYPE = ('.json', '.log', '.py',
                                                'yaml'),
                     keep_local: bool = True,
                     file_client_args: Optional[dict] = None,
                     log_metric_by_epoch: bool = True,
                     backend_args: Optional[dict] = None,
                     filtering_key: str = 'val'):

            super().__init__(interval, ignore_last, interval_exp_name, out_dir,
                             out_suffix, keep_local, file_client_args,
                             log_metric_by_epoch, backend_args)
            self.filtering_key = filtering_key

        def after_train_iter(self, runner: Runner, **kwargs) -> None:
            """Log after train itr.

            Args:
                runner (:obj:`mmengine.runner.Runner`): The runner to log.
            """
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
                # `runner.max_iters` may not be divisible by `self.interval`.
                # if `self.ignore_last==True`, the log of remaining iterations
                # will be recorded (Epoch [4][1000/1007], the logs of 998-1007
                # iterations will be recorded).
                tag, log_str = runner.log_processor.get_log_after_iter(
                    runner, batch_idx, 'train')
            else:
                return
            # runner.logger.info(log_str)

            # TODO: Here we sohuld feed tags to ray reporter

            rank, world_size = get_dist_info()
            if world_size > 1:
                if rank == 0:
                    broadcasted = [tag]
                else:
                    broadcasted = [None]
                dist.broadcast_object_list(broadcasted)
                tag = broadcasted.pop()
            if not any(
                    filter(lambda elem: self.filtering_key in elem,
                           tag.keys())):
                return
            tag['global_step'] = runner.iter + 1
            session.report(tag)

        def after_val_epoch(
                self,
                runner,
                metrics: Optional[Dict[str, float]] = None) -> None:
            """All subclasses should override this method, if they need any
            operations after each validation epoch.

            Args:
                runner (Runner): The runner of the validation process.
                metrics (Dict[str, float], optional): Evaluation results of all
                    metrics on validation dataset. The keys are the names of
                    the metrics, and the values are corresponding results.
            """
            tag, log_str = runner.log_processor.get_log_after_epoch(
                runner, len(runner.val_dataloader), 'val')
            tag = {k: v for k, v in tag.items() if 'time' not in k}

            rank, world_size = get_dist_info()
            if world_size > 1:
                if rank == 0:
                    broadcasted = [tag]
                else:
                    broadcasted = [None]
                dist.broadcast_object_list(broadcasted)
                tag = broadcasted.pop()

            session.report(metrics=tag)

else:
    from mmcv.runner import HOOKS
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

        def after_train_iter(self, runner: Runner) -> None:
            """Log after train itr.

            Args:
                runner (:obj:`mmengine.runner.Runner`): The runner to log.
            """
            if self.by_epoch and self.every_n_inner_iters(
                    runner, self.interval):
                runner.log_buffer.average(self.interval)
            elif not self.by_epoch and self.every_n_iters(
                    runner, self.interval):
                runner.log_buffer.average(self.interval)
            elif self.end_of_epoch(runner) and not self.ignore_last:
                # not precise but more stable
                runner.log_buffer.average(self.interval)

            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

        def after_train_epoch(self, runner: Runner) -> None:
            """Log after train epoch.

            Args:
                runner (:obj:`mmengine.runner.Runner`): The runner to log.
            """
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

        def after_val_epoch(self, runner: Runner) -> None:
            """Log after val epoch.

            Args:
                runner (:obj:`mmengine.runner.Runner`): The runner to log.
            """
            runner.log_buffer.average()
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

        def log(self, runner: Runner) -> None:
            """Log the information.

            Args:
                runner (:obj:`mmengine.runner.Runner`): The runner to log.
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
                    filter(lambda elem: self.filtering_key in elem,
                           tags.keys())):
                return
            tags['global_step'] = self.get_iter(runner)
            session.report(tags)
