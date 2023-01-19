# Copyright (c) SI-Analytics. All rights reserved.
from importlib.machinery import SourceFileLoader
from typing import Sequence

from siatune.utils import get_train_script
from .builder import TASKS
from .mm import MMBaseTask


class _EntrypointExecutor:
    """Execute the entrypoint of open mm train-based projects.

    Args:
        pkg_name (str): The abbreviation of the package.
        argv (Sequence[str]): The arguments for `tools/train.py`
        module_name (str):
            The name of the module to execute. Defaults to 'main'.
    """

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
        """Hijack the command line arguments.

        Args:
            argv (Sequence[str]): The arguments for `tools/train.py`
        """
        import sys
        sys.argv[1:] = argv
        return

    def execute(self):
        """Run the task."""
        self._hijack_argv(self._argv)
        getattr(self._entrypoint, self._module_name)()


@TASKS.register_module()
class MIM(MMBaseTask):
    """Wrapper class execute any script provided by all OpenMMLab codebases.

    Args:
        pkg_name (str): The abbreviation of the package.
    """

    def __init__(self, pkg_name: str, **kwargs):
        self._pkg_name = pkg_name
        super().__init__(should_parse=False, **kwargs)

    def parse_args(self, *args, **kwargs) -> None:
        pass

    def run(self, args: Sequence[str]):
        """This method runs a task in the MIM framework.

        Args:
            args (Sequence[str]): A list of command-line arguments.
        """
        executor = _EntrypointExecutor(self._pkg_name, args)
        executor.execute()
