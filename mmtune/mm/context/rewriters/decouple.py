from typing import List

from mmtune.utils import ImmutableContainer
from .builder import REWRITERS


@REWRITERS.register_module()
class Decouple:

    def __init__(self, key: str):
        self.key = key

    def __call__(self, context: dict):
        context[self.key] = ImmutableContainer.decouple(context[self.key])
