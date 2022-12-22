# Copyright (c) SI-Analytics. All rights reserved.
from typing import List, Tuple

def ref_raw_args(raw_args: List[str], key: str) -> Tuple:
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