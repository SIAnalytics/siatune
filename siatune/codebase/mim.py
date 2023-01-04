# Copyright (c) SI-Analytics. All rights reserved.
from typing import Sequence

from .builder import TASKS
from .mm import MMBaseTask

from importlib.machinery import SourceFileLoader
from typing import Sequence

from siatune.utils import get_train_script


class _EntrypointRunner:

    def __init__(self,
                 pkg_name: str,
                 argv: Sequence[str],
                 module_name: str = 'main'):
        self._train_script = get_train_script(pkg_name)
        self._module_name = module_name
        self._argv = argv
        self._entrypoint = SourceFileLoader(self._module_name,
                                            self._train_script).load_module()

    def _hijack_argv(self, argv: Sequence[str]):
        import sys
        sys.argv[1:] = argv
        return

    def run(self):
        self._hijack_argv(self._argv)
        getattr(self._entrypoint, self._module_name)()


@TASKS.register_module()
class MIM(MMBaseTask):

    def __init__(self, pkg_name: str, **kwargs):
        self._pkg_name = pkg_name
        super().__init__(should_parse=False, **kwargs)

    def parse_args(self, *args, **kwargs) -> None:
        return None

    def execute(self, args: Sequence[str]):
        runner = _EntrypointRunner(self._pkg_name, args)
        runner.run()
