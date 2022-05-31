import argparse
from abc import ABCMeta, abstractmethod
from copy import deepcopy
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

    def context_aware_run(self, *searched_cfg, **context) -> None:
        context_manager = ContextManager(self.rewriters)
        cp_context = dict(
            args=deepcopy(self.args),
            searched_cfg=deepcopy(searched_cfg[0]),
        )
        cp_context.update(context)
        return context_manager(self.run)(**cp_context)

    @abstractmethod
    def run(self, *, args=None, check) -> None:
        pass

    @abstractmethod
    def create_trainable(self, *args, **kwargs) -> ray.tune.Trainable:
        pass
