# Copyright (c) SI-Analytics. All rights reserved.
from typing import Callable, Sequence

from ray.tune import with_resources as reserve_resources

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
        self.dist_creator = DataParallelTrainCreator(
            self._run,
            num_cpus_per_worker=self.num_cpus_per_worker,
            num_workers=self.num_workers)

    def parse_args(self, *args, **kwargs) -> None:
        return None

    def run(self, *arg, **kwargs):
        self.dist_creator.train(*arg, **kwargs)

    def _run(self, *, args: Sequence[str], **kwargs) -> None:
        runner = EntrypointRunner(self._pkg_name, args)
        runner.run()

    def create_trainable(self) -> Callable:
        """Get ray trainable task.

        Returns:
            Callable: The Ray trainable task.
        """
        return reserve_resources(
            self.context_aware_run,
            self.dist_creator.resources,
        )
