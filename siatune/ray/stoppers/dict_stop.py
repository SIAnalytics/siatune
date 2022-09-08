# Copyright (c) SI-Analytics. All rights reserved.
from .builder import STOPPERS


@STOPPERS.register_module()
class DictionaryStopper(dict):
    """Dictionary type stopper."""
    pass
