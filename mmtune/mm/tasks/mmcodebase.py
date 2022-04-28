import argparse
from abc import abstractmethod
from typing import Optional

import mmcv
import ray
import torch

from .base import BaseTask


class MMTrainBasedTask(BaseTask):
    @staticmethod
    @abstractmethod
    def build_model(cfg: mmcv.Config, **kwargs) -> torch.nn.Module:
        pass

    @staticmethod
    @abstractmethod
    def build_dataset(cfg: mmcv.Config, **kwargs) -> torch.utils.data.Dataset:
        pass

    @staticmethod
    @abstractmethod
    def train_model(model: torch.nn.Moudle, dataset: torch.utils.data.Dataset, cfg: mmcv.Config, **kwargs) -> None:
        pass

    @staticmethod
    @abstractmethod
    def parse_args(parser: Optional[argparse.ArgumentParser]) -> argparse.ArgumentParser:
        pass

    @staticmethod
    @abstractmethod
    def run(**kwargs) -> None:
        pass


    @staticmethod
    @abstractmethod
    def create_torch_ddp_trainable(cfg) -> ray.tune.Trainable:
        pass
