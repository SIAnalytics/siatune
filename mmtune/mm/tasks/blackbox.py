import argparse
from typing import Callable, Optional

from .base import BaseTask
from .builder import TASKS


@TASKS.register_module()
class BloackBoxTask(BaseTask):

    @staticmethod
    def add_arguments(
        parser: Optional[argparse.ArgumentParser] = None
    ) -> argparse.ArgumentParser:

        if parser is None:
            parser = argparse.ArgumentParser(description='Train a segmentor')
        return parser

    @staticmethod
    def create_trainable() -> Callable:
        return BloackBoxTask.run
