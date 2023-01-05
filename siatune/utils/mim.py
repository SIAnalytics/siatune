# Copyright (c) SI-Analytics. All rights reserved.
from os import path as osp

from mim.utils import get_installed_path, is_installed, module_full_name


def get_train_script(pkg_name: str) -> str:
    pkg_full_name = module_full_name(pkg_name)
    if pkg_full_name == '':
        raise ValueError("Can't determine a unique "
                         f'package given abbreviation {pkg_name}')
    if not is_installed(pkg_full_name):
        raise RuntimeError(f'The codebase {pkg_name} is not installed')
    pkg_root = get_installed_path(pkg_full_name)
    # tools will be put in package/.mim in PR #68
    train_script = osp.join(pkg_root, '.mim', 'tools', 'train.py')
    if not osp.exists(train_script):
        _alternative = osp.join(pkg_root, 'tools', 'train.py')
        if osp.exists(_alternative):
            train_script = _alternative
        else:
            raise RuntimeError("Can't find train script")
    return train_script
