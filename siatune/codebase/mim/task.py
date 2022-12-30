# Copyright (c) SI-Analytics. All rights reserved.
from typing import Callable, Sequence

from siatune.core import DataParallelTrainCreator
from ..base import BaseTask
from ..builder import TASKS
from ._entrypoint import EntrypointRunner


@TASKS.register_module()
class MIM(BaseTask):

    def __init__(self, pkg_name: str, **kwargs):
        self._pkg_name = pkg_name
        super().__init__(should_parse=False, **kwargs)
        assert self.num_gpus_per_worker == 1

    def parse_args(self, *args, **kwargs) -> None:
        return None

    def run(self, *, args: Sequence[str], **kwargs) -> None:
        runner = EntrypointRunner(self._pkg_name, args)
        runner.run()

    def create_trainable(self) -> Callable:
        """Get ray trainable task.

        Returns:
            Callable: The Ray trainable task.
        """
        return DataParallelTrainCreator(
            self.context_aware_run,
            num_cpus_per_worker=self.num_cpus_per_worker,
            num_workers=self.num_workers).create()
