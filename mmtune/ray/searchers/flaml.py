from tune.search.flaml import CFO as _CFO
from tune.search.flaml import BlendSearch as _BlendSearch

from .builder import SEARCHERS


@SEARCHERS.register_module()
class BlendSearch(_BlendSearch):
    __doc__ = _BlendSearch.__doc__


@SEARCHERS.register_module()
class CFOSearch(_CFO):
    __doc__ = _CFO.__doc__
