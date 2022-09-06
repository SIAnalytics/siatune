# Copyright (c) SI-Analytics. All rights reserved.
import argparse
from abc import ABCMeta
from typing import Callable, Sequence

from .base import BaseTask
from .builder import TASKS


@TASKS.register_module()
class BlackBoxTask(BaseTask, metaclass=ABCMeta):
    """A general wrapping class for optimizing black box systems (without any
    knowledge of its internal workings.

    Its implementation is "opaque").
    """

    def parse_args(self, args: Sequence[str]) -> argparse.Namespace:
        """Define and parse the necessary arguments for the task.

        Args:
            args (Sequence[str]): The args.
        Returns:
            argparse.Namespace: The parsed args.
        """

        parser = argparse.ArgumentParser(description='black box')
        return parser.parse_args(args)

    def create_trainable(self) -> Callable:
        """Get ray trainable task.

        Returns:
            Callable: The Ray trainable task.
        """

        return self.context_aware_run
