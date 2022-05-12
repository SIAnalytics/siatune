from typing import List

from mmtune.utils import ImmutableContainer
from .builder import REWRITERS


@REWRITERS.register_module()
class Decouple:

    def __init__(self, keys: List[str] = []):
        self.keys = keys

    def __call__(self, context: dict):
        for key in self.keys:
            context[key] = ImmutableContainer.decouple(context[key])
        return context
