import argparse
from functools import partial
from typing import Callable, Optional

from .base import BaseTask
from .builder import TASKS


@TASKS.register_module()
class BloackBoxTask(BaseTask):

    @classmethod
    def add_arguments(
        cls,
        parser: Optional[argparse.ArgumentParser] = None
    ) -> argparse.ArgumentParser:

        if parser is None:
            parser = argparse.ArgumentParser(description='Train a segmentor')
        return parser

    @classmethod
    def create_trainable(cls) -> Callable:
        return partial(
            cls.contextaware_run,
            dict(
                base_cfg=BloackBoxTask.BASE_CFG,
                args=BloackBoxTask.ARGS,
                rewriters=BaseTask.REWRITERS))
