import argparse
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

import ray

from mmtune.mm.context import ContextManager


class BaseTask(metaclass=ABCMeta):
    """Wrap the apis of target task."""

    def __init__(self, rewriters: List[dict] = []) -> None:
        """Initialize the task.

        Args:
            rewriters (List[dict]):
                Context redefinition pipeline. Defaults to [].
        """

        self._args: Optional[argparse.Namespace] = None
        self._rewriters: List[dict] = rewriters

    def set_args(self, args: Sequence[str]) -> None:
        """Parse and Set the args.

        Args:
            args (Sequence[str]): The args.
        """

        self._args = self.parse_args(args)

    @property
    def args(self) -> argparse.Namespace:
        return self._args

    @property
    def rewriters(self) -> List[dict]:
        return self._rewriters

    @abstractmethod
    def parse_args(self, args: Sequence[str]) -> argparse.Namespace:
        """Define and parse the necessary arguments for the task.

        Args:
            args (Sequence[str]): The args.
        Returns:
            argparse.Namespace: The parsed args.
        """
        pass

    def context_aware_run(self, searched_cfg: Dict, **kwargs) -> Any:
        """Gathers and refines the information received by users and Raytune to
        execute the objective task.

        Args:
            searched_cfg (Dict): The searched configuration.
            kwargs (**kwargs): The kwargs.
        Returns:
            Any: The result of the objective task.
        """

        context_manager = ContextManager(self.rewriters)
        context = dict(
            args=deepcopy(self.args),
            searched_cfg=deepcopy(searched_cfg),
        )
        context.update(kwargs)
        return context_manager(self.run)(**context)

    @abstractmethod
    def run(self, *, args: argparse.Namespace, **kwargs) -> None:
        """The objective task.

        Args:
            args (argparse.Namespace): The args.
            kwargs (Dict): The kwargs.
        """
        pass

    @abstractmethod
    def create_trainable(self, *args, **kwargs) -> ray.tune.Trainable:
        """Get ray trainable task.

        Args:
            args (argparse.Namespace): The args.
            kwargs (Dict): The kwargs.
        """
        pass
