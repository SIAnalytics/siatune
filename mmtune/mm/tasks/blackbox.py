import argparse
from abc import ABCMeta
from functools import partial
from typing import Callable, Sequence

from .base import BaseTask
from .builder import TASKS


@TASKS.register_module()
class BloackBoxTask(BaseTask, metaclass=ABCMeta):

    def parse_args(self, args: Sequence[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='black box')
        return parser.parse_args(args)

    def create_trainable(self) -> Callable:
        return partial(
            self.context_aware_run,
            dict(
                base_cfg=self.base_cfg,
                args=self.args,
                rewriters=self.rewriters))
