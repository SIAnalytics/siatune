# Copyright (c) SI-Analytics. All rights reserved.
from typing import List, Tuple


def reference_raw_args(raw_args: List[str], key: str) -> Tuple:
    """Extract the list of arguments and their indices from a list of command-
    line arguments.

    The list of arguments is considered
    to start after the first occurrence of the given key
    and to end at the next occurrence
    of a key (a string starting with '--').

    Args:
        raw_args (List[str]): List of command-line arguments.
        key (str):
            Key to look for in the raw arguments.
            The key should start with '--'.

    Returns:
        Tuple[List[str], List[int]]: The list of arguments and their indices.
    """
    assert key.startswith('--')
    ret: List[str] = []
    ret_indices: List[int] = []
    if key not in raw_args:
        return ret, ret_indices
    for idx in range(raw_args.index(key) + 1, len(raw_args)):
        cand = raw_args[idx]
        if cand.startswith('--'):
            break
        ret.append(cand)
        ret_indices.append(idx)
    return ret, ret_indices
