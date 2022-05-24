from typing import Dict, List

from mmtune.utils import ImmutableContainer
from .builder import REWRITERS


@REWRITERS.register_module()
class Decouple:

    def __init__(self, keys: List[str] = []):
        self.keys = keys

    def __call__(self, context: Dict):
        assert set(context).issuperset(set(
            self.keys)), ('context should have superset of keys!')

        for key in self.keys:
            context[key] = ImmutableContainer.decouple(context[key])
        return context
