import argparse
from abc import ABCMeta
from typing import Callable, Sequence

from .base import BaseTask
from .builder import TASKS


@TASKS.register_module()
class BlackBoxTask(BaseTask, metaclass=ABCMeta):

    def parse_args(self, args: Sequence[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='black box')
        return parser.parse_args(args)

    def create_trainable(self) -> Callable:
        return self.context_aware_run
