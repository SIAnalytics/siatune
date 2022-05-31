import argparse
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import List, Optional, Sequence

import ray

from mmtune.mm.context import ContextManager
from .builder import TASKS


@TASKS.register_module()
class BaseTask(metaclass=ABCMeta):
    """Wrap the apis of target task."""

    def __init__(self, rewriters: List[dict] = []):
        self._args: Optional[argparse.Namespace] = None
        self._rewriters: List[dict] = rewriters

    def set_args(self, args: Sequence[str]) -> None:
        self._args = self.parse_args(args)

    @property
    def args(self) -> argparse.Namespace:
        return self._args

    @property
    def rewriters(self) -> List[dict]:
        return self._rewriters

    @abstractmethod
    def parse_args(self, args: Sequence[str]) -> argparse.Namespace:
        pass

    def context_aware_run(self, searched_cfg, **context) -> None:
        context_manager = ContextManager(self.rewriters)
        cp_context = dict(
            args=deepcopy(self.args),
            searched_cfg=deepcopy(searched_cfg),
        )
        cp_context.update(context)
        return context_manager(self.run)(**cp_context)

    @abstractmethod
    def run(self, *, args, **kwargs) -> None:
        pass

    @abstractmethod
    def create_trainable(self, *args, **kwargs) -> ray.tune.Trainable:
        pass
