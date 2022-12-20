# Copyright (c) SI-Analytics. All rights reserved.
import re
from typing import Callable, Optional, Union

import ray.tune as tune
from ray.tune.search.sample import Domain

from .base import BaseSpace
from .builder import SPACES


@SPACES.register_module()
class SampleFrom(BaseSpace):
    """Specify that tune should sample configuration values from this function.

    Args:
        func (str | Callable): An string or callable function
            to draw a sample from.
    """

    sample: Callable = tune.sample_from

    def __init__(self,
                 func: Union[str, Callable],
                 imports: Optional[list] = None) -> None:
        if isinstance(func, str):
            assert func.startswith('lambda')
        imports = imports or []

        head, *expr = re.split(r':', func)
        self.func = eval(head + ': exec("' + ';'.join(f'import {m}'
                                                      for m in imports) +
                         '") or ' + ':'.join(expr))

    @property
    def space(self) -> Domain:
        return self.sample.__func__(self.func)
