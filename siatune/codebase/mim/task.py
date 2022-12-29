# Copyright (c) SI-Analytics. All rights reserved.
import argparse
from typing import Callable, Sequence

from siatune.core import create_dist_trainer
from ..base import BaseTask
from ..builder import TASKS
from ._entrypoint import EntrypointRunner


@TASKS.register_module()
class MIM(BaseTask):

    def __init__(self, pkg_name: str, **kwargs):
        self._pkg_name = pkg_name
        super().__init__(**kwargs)
        assert self.num_gpus_per_worker == 1

    def parse_args(self, args: Sequence[str]) -> argparse.Namespace:
        return argparse.Namespace()

    def run(self, *, raw_args: Sequence[str], **kwargs) -> None:
        runner = EntrypointRunner(self._pkg_name, raw_args)
        runner.run()

    def create_trainable(self) -> Callable:
        """Get ray trainable task.

        Returns:
            Callable: The Ray trainable task.
        """
        return create_dist_trainer(
            self.context_aware_run,
            num_cpus_per_worker=self.num_cpus_per_worker,
            num_workers=self.num_workers)
