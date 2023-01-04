# Copyright (c) SI-Analytics. All rights reserved.
from typing import Sequence

from ..builder import TASKS
from ..mm import MMBaseTask
from ._entrypoint import EntrypointRunner


@TASKS.register_module()
class MIM(MMBaseTask):

    def __init__(self, pkg_name: str, **kwargs):
        self._pkg_name = pkg_name
        super().__init__(should_parse=False, **kwargs)

    def parse_args(self, *args, **kwargs) -> None:
        return None

    def train(self, args: Sequence[str]):
        runner = EntrypointRunner(self._pkg_name, args)
        runner.run()
