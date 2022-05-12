import argparse
from abc import ABCMeta, abstractmethod
from typing import List, Optional

import ray
from mmcv.utils.config import Config

from mmtune.mm.context import ContextManager
from mmtune.utils import ImmutableContainer
from .builder import TASKS


@TASKS.register_module()
class BaseTask(metaclass=ABCMeta):
    """Wrap the apis of target task."""
    BASE_CFG: Optional[ImmutableContainer] = None
    ARGS: Optional[argparse.Namespace] = None
    REWRITERS: List[dict] = []

    @classmethod
    def set_base_cfg(cls, base_cfg: Config) -> None:
        BaseTask.BASE_CFG = ImmutableContainer(base_cfg, 'base')

    @classmethod
    def set_args(cls, args: argparse.Namespace) -> None:
        BaseTask.ARGS = args

    @classmethod
    def set_rewriters(cls, rewriters: List[dict] = []) -> None:
        BaseTask.REWRITERS = rewriters

    @classmethod
    @abstractmethod
    def add_arguments(
        cls,
        parser: Optional[argparse.ArgumentParser] = None
    ) -> argparse.ArgumentParser:
        pass

    @classmethod
    def contextaware_run(cls, status, *args, **kwargs) -> None:
        context_manager = ContextManager(**status)
        return context_manager(cls.run)(*args, **kwargs)

    @classmethod
    @abstractmethod
    def run(cls, *args, **kwargs) -> None:
        pass

    @classmethod
    @abstractmethod
    def create_trainable(cls, *args, **kwargs) -> ray.tune.Trainable:
        pass
