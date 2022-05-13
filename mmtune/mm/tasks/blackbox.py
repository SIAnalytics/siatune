import argparse
from abc import ABCMeta
from functools import partial
from typing import Callable, Optional

from .base import BaseTask
from .builder import TASKS


@TASKS.register_module()
class BloackBoxTask(BaseTask, metaclass=ABCMeta):

    def add_arguments(
        self,
        parser: Optional[argparse.ArgumentParser] = None
    ) -> argparse.ArgumentParser:

        if parser is None:
            parser = argparse.ArgumentParser(description='black box')
        return parser

    def create_trainable(self) -> Callable:
        return partial(
            self.contextaware_run,
            dict(
                base_cfg=self.BASE_CFG,
                args=self.ARGS,
                rewriters=self.REWRITERS))
