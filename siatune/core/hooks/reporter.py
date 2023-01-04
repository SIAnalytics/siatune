# Copyright (c) SI-Analytics. All rights reserved.
import time
from pathlib import Path
from typing import Dict, Optional, Union

from ray.air import session
from ray.air.checkpoint import Checkpoint

from siatune.version import IS_DEPRECATED_MMCV

if not IS_DEPRECATED_MMCV:
    import mmengine
    from mmengine.dist import master_only
    from mmengine.hooks import LoggerHook
    from mmengine.hooks.logger_hook import SUFFIX_TYPE
    from mmengine.model import is_model_wrapper
    from mmengine.optim import OptimWrapper
    from mmengine.registry import HOOKS
    from mmengine.runner import Runner
    from mmengine.runner.checkpoint import get_state_dict, weights_to_cpu
    from mmengine.utils import get_git_hash

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
                     filtering_key: str = 'val',
                     with_ckpt: bool = True):

            super().__init__(interval, ignore_last, interval_exp_name, out_dir,
                             out_suffix, keep_local, file_client_args,
                             log_metric_by_epoch, backend_args)
            self.filtering_key = filtering_key
            self.with_ckpt = with_ckpt

        @master_only
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
                tag, _ = runner.log_processor.get_log_after_iter(
                    runner, batch_idx, 'train')
            elif (self.end_of_epoch(runner.train_dataloader, batch_idx)
                  and not self.ignore_last):
                # `runner.max_iters` may not be divisible by `self.interval`.
                # if `self.ignore_last==True`, the log of remaining iterations
                # will be recorded (Epoch [4][1000/1007], the logs of 998-1007
                # iterations will be recorded).
                tag, _ = runner.log_processor.get_log_after_iter(
                    runner, batch_idx, 'train')
            else:
                return
            # runner.logger.info(log_str)
            if not any(
                    filter(lambda elem: self.filtering_key in elem,
                           tag.keys())):
                return
            # TODO: Here we sohuld feed tags to ray reporter
            if self.with_ckpt:
                session.report(tag, Checkpoint=self._save_checkpoint(runner))
            else:
                session.report(tag)

        @master_only
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
            tag, _ = runner.log_processor.get_log_after_epoch(
                runner, len(runner.val_dataloader), 'val')
            tag = {k: v for k, v in tag.items() if 'time' not in k}
            if not any(
                    filter(lambda elem: self.filtering_key in elem,
                           tag.keys())):
                return
            if self.with_ckpt:
                session.report(
                    tag, Checkpoint=self._save_checkpoint(runner, False))
            else:
                session.report(tag)

        def _save_checkpoint(self, runner: Runner, is_train: bool) -> None:
            """Save checkpoints periodically.

            Args:
                runner (:obj:`mmcv.runner.BaseRunner`):
                    The runner to save checkpoints.
            """
            itr = runner.iter
            if is_train:
                itr += 1

            meta = dict()
            meta.update(
                cfg=runner.cfg.pretty_text,
                seed=runner.seed,
                experiment_name=runner.experiment_name,
                time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
                mmengine_version=mmengine.__version__ + get_git_hash(),
                epoch=runner.epoch,
                iter=itr)

            if hasattr(runner.train_dataloader.dataset, 'metainfo'):
                meta.update(
                    dataset_meta=runner.train_dataloader.dataset.metainfo)

            if is_model_wrapper(runner.model):
                model = runner.model.module
            else:
                model = runner.model

            checkpoint = {
                'meta': meta,
                'state_dict': weights_to_cpu(get_state_dict(model)),
                'message_hub': runner.message_hub.state_dict()
            }
            # save optimizer state dict to checkpoint
            if isinstance(runner.optim_wrapper, OptimWrapper):
                checkpoint['optimizer'] = runner.optim_wrapper.state_dict()
            else:
                raise TypeError(
                    'runner.optim_wrapper should be an `OptimWrapper` '
                    'or `OptimWrapperDict` instance, but got '
                    f'{runner.optim_wrapper}')

            # save param scheduler state dict
            if isinstance(runner.param_schedulers, dict):
                checkpoint['param_schedulers'] = dict()
                for name, schedulers in runner.param_schedulers.items():
                    checkpoint['param_schedulers'][name] = []
                    for scheduler in schedulers:
                        state_dict = scheduler.state_dict()
                        checkpoint['param_schedulers'][name].append(state_dict)
            else:
                checkpoint['param_schedulers'] = []
                for scheduler in runner.param_schedulers:  # type: ignore
                    state_dict = scheduler.state_dict()  # type: ignore
                    checkpoint['param_schedulers'].append(state_dict)
            return Checkpoint.from_dict(checkpoint)

else:
    import mmcv
    from mmcv.parallel import is_module_wrapper
    from mmcv.runner import HOOKS, BaseRunner
    from mmcv.runner.checkpoint import get_state_dict, weights_to_cpu
    from mmcv.runner.dist_utils import master_only
    from mmcv.runner.hooks.logger import LoggerHook
    from torch.optim import Optimizer

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
            with_ckpt: bool = True,
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
            self.with_ckpt = with_ckpt

        def after_train_iter(self, runner: BaseRunner) -> None:
            """Log after train itr.

            Args:
                runner (:obj:`BaseRunner`): The runner to log.
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

        def after_train_epoch(self, runner: BaseRunner) -> None:
            """Log after train epoch.

            Args:
                runner (:obj:`BaseRunner`): The runner to log.
            """
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

        def after_val_epoch(self, runner: BaseRunner) -> None:
            """Log after val epoch.

            Args:
                runner (:obj:`BaseRunner`): The runner to log.
            """
            runner.log_buffer.average()
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

        @master_only
        def log(self, runner: BaseRunner) -> None:
            """Log the information.

            Args:
                runner (:obj:`BaseRunner`): The runner to log.
            """

            tags = self.get_loggable_tags(runner)
            if not any(
                    filter(lambda elem: self.filtering_key in elem,
                           tags.keys())):
                return
            if self.with_ckpt:
                session.report(tags, checkpoint=self._save_checkpoint(runner))
            else:
                session.report(tags)

        def _save_checkpoint(self, runner: BaseRunner) -> Checkpoint:
            model = runner.model
            optimizer = runner.optimizer

            meta = dict(
                mmcv_version=mmcv.__version__,
                time=time.asctime(),
                epoch=runner.epoch + 1,
                iter=runner.iter + 1)
            if is_module_wrapper(model):
                model = model.module
            if hasattr(model, 'CLASSES') and model.CLASSES is not None:
                # save class name to the meta
                meta.update(CLASSES=model.CLASSES)
            checkpoint = {
                'meta': meta,
                'state_dict': weights_to_cpu(get_state_dict(model))
            }
            if isinstance(optimizer, Optimizer):
                checkpoint['optimizer'] = optimizer.state_dict()
            elif isinstance(optimizer, dict):
                checkpoint['optimizer'] = {}
                for name, optim in optimizer.items():
                    checkpoint['optimizer'][name] = optim.state_dict()
            return Checkpoint.from_dict(checkpoint)
