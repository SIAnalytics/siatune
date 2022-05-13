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

    def __init__(self):
        self.base_cfg: Optional[ImmutableContainer] = None
        self.args: Optional[argparse.Namespace] = None
        self.rewriters: List[dict] = []

    def set_base_cfg(self, base_cfg: Config) -> None:
        self.base_cfg = ImmutableContainer(base_cfg, 'base')

    def set_args(self, args: argparse.Namespace) -> None:
        self.args = args

    def set_rewriters(self, rewriters: List[dict] = []) -> None:
        self.rewriters = rewriters

    @abstractmethod
    def add_arguments(
        self,
        parser: Optional[argparse.ArgumentParser] = None
    ) -> argparse.ArgumentParser:
        pass

    def contextaware_run(self, status, *args, **kwargs) -> None:
        context_manager = ContextManager(**status)
        return context_manager(self.run)(*args, **kwargs)

    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def create_trainable(self, *args, **kwargs) -> ray.tune.Trainable:
        pass
