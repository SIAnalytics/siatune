# Copyright (c) SI-Analytics. All rights reserved.
from importlib.machinery import SourceFileLoader
from typing import Sequence

from siatune.utils import get_train_script


class EntrypointRunner:

    def __init__(self,
                 pkg_name: str,
                 argv: Sequence[str],
                 module_name: str = 'main'):
        self._train_script = get_train_script(pkg_name)
        self._module_name = module_name
        self._argv = argv
        self._entrypoint = SourceFileLoader(self._module_name,
                                            self._train_script).load_module()

    def _hijack_argv(self, argv):
        import sys
        sys.argv[1:] = argv
        return

    def run(self):
        self._hijack_argv(self._argv)
        getattr(self._entrypoint, self._module_name)()
