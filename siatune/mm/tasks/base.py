# Copyright (c) SI-Analytics. All rights reserved.
import argparse
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

import ray

from siatune.mm.context import ContextManager
from siatune.utils import ImmutableContainer


class BaseTask(metaclass=ABCMeta):
    """Base class to specify the target task.

    The main functions of the task processor are as follows:
    1. CLI argument definition and parsing. (`parse_args`)
        Inputs: args (Sequence[str])
        Outputs: args (argparse.Namespace)
    2. Objective function execution.
        The result must be reported by calling tune.report. (`run`)
        Inputs: args (argparse.Namespace)
        Outputs: None
    3. Gather and refine information from multiple sources.
        Call the 'run' function of the task processor. (`context_aware_run`)
        Aggregate the information we define as context,
        convert it into a refined argparse namespace, and input it to run.
        The context consists of:
            1. args (argparse.Namespace): The low level CLI arguments.
            2. searched_cfg (Dict):
                The configuration searched by the algorithm.
            3. checkpoint_dir (Optional[str]):
                The directory of checkpoints that contains the states.
        Inputs: searched_cfg (Dict), checkpoint_dir (Optional[str])
        Outputs: None
    """

    def __init__(self, rewriters: List[dict] = []) -> None:
        """Initialize the task.

        Args:
            rewriters (List[dict]):
                Context redefinition pipeline. Defaults to [].
        """

        self._args: Optional[argparse.Namespace] = None
        self._rewriters: List[dict] = rewriters

    def set_args(self, args: Sequence[str]) -> None:
        """Parse and set the argss.

        Args:
            args (Sequence[str]): The args.
        """

        self._args = self.parse_args(args)

    def set_resource(self,
                     num_cpus_per_worker: int = 1,
                     num_gpus_per_worker: int = 1,
                     num_workers: int = 1) -> None:
        """Set resource per trial.

        Args:
            num_cpus_per_worker (int):
                The number of CPUs that one worker can use. Defaults to 1.
            num_gpus_per_worker (int):
                The number of GPUs that one worker can use. Defaults to 1.
            num_workers (int): The number of workers. Defaults to 1.
        """
        assert num_workers > 0
        self._resource = dict(
            num_cpus_per_worker=num_cpus_per_worker,
            num_gpus_per_worker=num_gpus_per_worker,
            num_workers=num_workers,
        )

    @property
    def num_cpus_per_worker(self) -> int:
        return self._resource.get('num_cpus_per_worker')

    @property
    def num_gpus_per_worker(self) -> int:
        return self._resource.get('num_gpus_per_worker')

    @property
    def num_workers(self) -> int:
        return self._resource.get('num_workers')

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

    def context_aware_run(self,
                          searched_cfg: Dict,
                          checkpoint_dir: Optional[str] = None,
                          **kwargs) -> Any:
        """Gather and refine the information received by users and Ray.tune to
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
            searched_cfg=deepcopy(ImmutableContainer.decouple(searched_cfg)),
            checkpoint_dir=checkpoint_dir,
        )
        context.update(kwargs)
        return context_manager(self.run)(**context)

    @abstractmethod
    def run(self, *, args: argparse.Namespace, **kwargs) -> None:
        """Run the target task.

        Args:
            args (argparse.Namespace): The args.
            kwargs (Dict): The kwargs.
        """
        pass

    @abstractmethod
    def create_trainable(self, *args, **kwargs) -> ray.tune.Trainable:
        """Get ray trainable task.

        Args:
            args (Tuple): The args.
            kwargs (Dict): The kwargs.
        """
        pass
