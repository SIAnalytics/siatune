import argparse
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence

import ray

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

    def set_args(self, args: Sequence[str]) -> None:
        self.args = self.parse_args(args)

    def set_rewriters(self, rewriters: List[dict] = []) -> None:
        self.rewriters = rewriters

    @abstractmethod
    def parse_args(self, args: Sequence[str]) -> argparse.Namespace:
        pass

    def contextaware_run(self, *searched_cfg, **context) -> None:
        context_manager = ContextManager(self.rewriters)
        context['args'] = self.args
        return context_manager(self.run)(*searched_cfg, **context)

    @abstractmethod
    def run(self, *, args=None) -> None:
        pass

    @abstractmethod
    def create_trainable(self, *args, **kwargs) -> ray.tune.Trainable:
        pass
