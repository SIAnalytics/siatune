# Copyright (c) SI-Analytics. All rights reserved.
import argparse
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Optional, Sequence, Union

from ray.tune import Trainable

from siatune.core import ContextManager
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
            1. args (argparse.Namespace | Sequence[str]):
                The low level CLI arguments.
            2. searched_cfg (Dict):
                The configuration searched by the algorithm.
        Inputs: searched_cfg (Dict)
        Outputs: None

    Args:
        args (Sequence[str]): The task arguments. It is parsed by
            :method:`parse_args`.
        num_workers (int): The number of workers to launch.
        num_cpus_per_worker (int): The number of CPUs per worker.
            Default to 1.
        num_gpus_per_worker (int): The number of GPUs per worker.
            Default to 1.
        rewriters (list[dict] | dict, optional): Context redefinition
            pipeline. Default to None.
    """

    def __init__(self,
                 args: Sequence[str],
                 num_workers: int = 1,
                 num_cpus_per_worker: int = 1,
                 num_gpus_per_worker: int = 1,
                 rewriters: Optional[Union[list, dict]] = None,
                 should_parse: bool = True):

        if should_parse:
            args = self.parse_args(args)
        self.args = args

        self.num_workers = num_workers
        self.num_cpus_per_worker = num_cpus_per_worker
        self.num_gpus_per_worker = num_gpus_per_worker

        if isinstance(rewriters, dict):
            rewriters = [rewriters]
        self.rewriters = rewriters

    @abstractmethod
    def parse_args(self,
                   args: Sequence[str]) -> Union[argparse.Namespace, None]:
        """Define and parse the necessary arguments for the task.

        Args:
            args (Sequence[str]): The args.

        Returns:
            argparse.Namespace: The parsed args.
        """
        pass

    def context_aware_run(self, searched_cfg: dict):
        """Gather and refine the information received by users and Ray.tune to
        execute the objective task.

        Args:
            searched_cfg (Dict): The searched configuration.
        """

        context_manager = ContextManager(self.rewriters)
        context = dict(
            args=deepcopy(self.args),
            searched_cfg=deepcopy(ImmutableContainer.decouple(searched_cfg)),
        )
        return context_manager(self.run)(**context)

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the trainable task."""
        pass

    @abstractmethod
    def create_trainable(self) -> Trainable:
        """Get ray trainable task."""
        pass
