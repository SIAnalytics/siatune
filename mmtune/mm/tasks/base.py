import argparse
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, List, Optional

import ray
from mmcv.utils.config import Config

from mmtune.mm.context import ContextManager
from mmtune.utils import ImmutableContainer
from .builder import TASK


@TASK.register_module()
class BaseTask(metaclass=ABCMeta):
    """Wrap the apis of target task."""
    BASE_CFG: Optional[ImmutableContainer] = None
    ARGS: Optional[argparse.Namespace] = None
    REWRITERS: List[dict] = []

    @staticmethod
    def set_base_cfg(base_cfg: Config) -> None:
        BaseTask.BASE_CFG = ImmutableContainer(base_cfg, 'base')

    @staticmethod
    def set_args(args: argparse.Namespace) -> None:
        BaseTask.ARGS = args

    @staticmethod
    def set_rewriters(rewriters: List[dict] = []) -> None:
        BaseTask.REWRITERS = rewriters

    @abstractmethod
    @staticmethod
    def add_arguments(
        parser: Optional[argparse.ArgumentParser] = None
    ) -> argparse.ArgumentParser:
        pass

    def _change_context(run: Callable):
        context_manager = ContextManager(BaseTask.BASE_CFG, BaseTask.ARGS,
                                         BaseTask.REWRITERS)
        return context_manager(run)

    @abstractmethod
    @staticmethod
    @_change_context
    def run(*args, **kwargs) -> Any:
        pass

    @abstractmethod
    @staticmethod
    def create_trainable(*args, **kwargs) -> ray.tune.Trainable:
        pass
