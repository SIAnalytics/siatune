# Copyright (c) SI-Analytics. All rights reserved.
import re
from typing import Dict, Tuple

from .base import BaseRewriter
from .builder import REWRITERS

WRAPPING_REGEXP = r'^\$\((.*)\)$'


def unwrap_regexp(value: str,
                  regexp: str = WRAPPING_REGEXP) -> Tuple[str, bool]:
    """Unwrap the value if it is wrapped by the regexp.

    Args:
        value (str): The value to unwrap.
        regexp (str): The regexp to match.
        Defaults to WRAPPING_REGEXP.

    Returns:
        str: The unwrapped value.
        bool: Whether the value is wrapped.
    """
    if not isinstance(value, str):
        return value, False
    searched = re.search(regexp, value)
    if searched:
        value = searched.group(1)
    return value, bool(searched)


@REWRITERS.register_module()
class BatchConfigPatcher(BaseRewriter):
    """Patch the config in the context. It was introduced with the intention of
    changing two or more elements at once. To take object detection as an
    example, suppose you want to find the optimal number of fpn output
    channels.

    'base cfg'
    model = dict(
        ...,
        neck=dict(
            type='FPN',
            ...,
            out_channels=x,
        )
        roi_head=dict(
            ...,
            bbox_head=dict(
                in_channels=b,
            )
        )
    )

    If the tuning algorithm provided by ray recommends ``y``,
    we need to change the neck and head channels
    from ``x`` to ``y`` at the same time.
    It is recommended to use this class in that case.
    The usage is as follows:

    'search space'
    {
        '$(model.neck.out_channels & model.roi_head.bbox_head.in_channels)':
        dict(type='Randint', lower=a, upper=b)
    }

    'context rewriter'
    dict(type='BatchConfigPatcher', key=...),

    1. Connect the elements you want
    to change collectively with ``&`` in the search configuration.
    2. Specifies the use of BatchConfigPatcher in the context writer.
    """

    def __init__(self, key: str) -> None:
        """Initialize the rewriter.

        Args:
            key (str): The key of the config in the context.
        """
        self.key = key

    def __call__(self, context: Dict) -> Dict:
        """Batch adjustment of elements connected by &.

        Args:
            context (Dict): The context to be rewritten.

        Returns:
            Dict: The context after rewriting.
        Examples:
            inputs = {'$(a & b)': 1}
            >>> BatchConfigPatcher()(inputs)
            {'a': 1, 'b': 1}
        """
        cfg = context[self.key]

        for key, value in cfg.copy().items():
            inner_key, is_wrapped = unwrap_regexp(key)
            if not is_wrapped:
                continue
            cfg.pop(key)
            if ' & ' in inner_key:
                for sub_key in inner_key.split(' & '):
                    cfg[sub_key] = value

        return context


@REWRITERS.register_module()
class SequeunceConfigPatcher(BaseRewriter):
    """Patch the config in the context. An extended form of BatchConfigPatcher.

     The change is that the elements to be changed
     in batches do not have to have the same value.
     To take semantic segmentation as an example,
     suppose you are looking for the optimal backbone
     feature level needed to make the final prediction.
     The size of the tensor from the backbone is equal to
     [torch.Size(b_0, c_0, h_0, w_0), torch.Size(b_1, c_1, h_1, w_1), ...].

     'base cfg'
     model = dict(
         type='EncoderDecoder',
         backbone=dict(
             ...,
         )
         decode_head=dict(
             type='ASPPHead',
             in_channels=c_x,
             in_index=x,
             ...,
         ),
         ...,
     )

     If you change the in_index of the decode head,
     you will also have to change the in_inchannels to construct a valid model.
     It is recommended to use this class in that case.
     If the ray tuning algorithm recommends ``y`` as the channel index,
     the number of channels also needs to be changed from ``c_x`` to ``c_y``.
     The usage is as follows:

    'search space'
     {
         '$(model.decode_head.in_channels - model.decode_head.in_index)' =
         dict(type='Choice', categories=[[c_a, a], [c_b, b], ...])
     }
     'context rewriter'
     dict(type='SequeunceConfigPatcher', key=...),

     1. Connect the elements you want
     to change collectively with ``-`` in the search configuration.
     2. Specifies the use of SequeunceConfigPatcher in the context writer.
    """

    def __init__(self, key: str) -> None:
        """Initialize the rewriter.

        Args:
            key (str): The key of the config path in the context.
        """

        self.key = key

    def __call__(self, context: Dict) -> Dict:
        """Adjust the elements connected by - in order.

        Args:
            context (Dict): The context to be rewritten.

        Returns:
            Dict: The context after rewriting.
        Examples:
            inputs = {'$(a - b)': [1, 2]}
            >>> SequeunceConfigPatcher()(inputs)
            {'a': 1, 'b': 2}
        """
        cfg = context[self.key]

        for key, value in cfg.copy().items():
            inner_key, is_wrapped = unwrap_regexp(key)
            if not is_wrapped:
                continue
            cfg.pop(key)
            if ' - ' in inner_key:
                for idx, sub_key in enumerate(inner_key.split(' - ')):
                    cfg[sub_key] = value[idx]

        return context
