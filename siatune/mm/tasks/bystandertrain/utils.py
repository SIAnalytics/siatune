from pkg_resources import get_distribution
from os import path as osp
import importlib
import pkg_resources

PKG2PROJECT = {
    'mmcv-full': 'mmcv',
    'mmcls': 'mmclassification',
    'mmdet': 'mmdetection',
    'mmdet3d': 'mmdetection3d',
    'mmsegmentation': 'mmsegmentation',
    'mmaction2': 'mmaction2',
    'mmtrack': 'mmtracking',
    'mmpose': 'mmpose',
    'mmedit': 'mmediting',
    'mmocr': 'mmocr',
    'mmgen': 'mmgeneration',
    'mmselfsup': 'mmselfsup',
    'mmrotate': 'mmrotate',
    'mmflow': 'mmflow',
    'mmyolo': 'mmyolo',
}

def get_installed_path(package: str) -> str:
    """Get installed path of package.
    Args:
        package (str): Name of package.
    Example:
        >>> get_installed_path('mmcls')
        >>> '.../lib/python3.7/site-packages/mmcls'
    """
    # if the package name is not the same as module name, module name should be
    # inferred. For example, mmcv-full is the package name, but mmcv is module
    # name. If we want to get the installed path of mmcv-full, we should concat
    # the pkg.location and module name
    pkg = get_distribution(package)
    possible_path = osp.join(pkg.location, package)
    if osp.exists(possible_path):
        return possible_path
    else:
        return osp.join(pkg.location, package2module(package))


def package2module(package: str):
    """Infer module name from package.
    Args:
        package (str): Package to infer module name.
    """
    pkg = get_distribution(package)
    if pkg.has_metadata('top_level.txt'):
        module_name = pkg.get_metadata('top_level.txt').split('\n')[0]
        return module_name
    else:
        raise ValueError(f'can not infer the module name of {package}')


def revert_args(args: dict) -> str:
    """Convert args dictionary to a string.
    Args:
        args (dict): A dictionary that contains parsed args.
    Return:
        A converted string.
    Example:
        >>> args = {
            'arg1': [value1, value2],
            'arg2': [value3],
            'arg3': [value4]
        }
        >>> revert_args(args)
        '--arg1 value1 value2 --arg2 value3 --arg3 value4'
    """
    result = []
    for key in args:
        revert_key = key.replace('_', '-')
        result.append(f'--{revert_key}')
        if isinstance(args[key], bool):
            pass        
        elif isinstance(args[key], list):
            result.extend([str(x) for x in args[key]])
        else:
            result.append(str(args[key]))
    if result[0] == '--conifg':
        result.pop(0)
    return result

def is_installed(package: str) -> bool:
    """Check package whether installed.
    Args:
        package (str): Name of package to be checked.
    """
    # refresh the pkg_resources
    # more datails at https://github.com/pypa/setuptools/issues/373
    importlib.reload(pkg_resources)
    try:
        get_distribution(package)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def module_full_name(abbr: str) -> str:
    """Get the full name of the module given abbreviation.
    Args:
        abbr (str): The abbreviation, should be the sub-string of one
            (and only one) supported module.
    Return:
        str: The full name of the corresponding module. If abbr is the
            sub-string of zero / multiple module names, return empty string.
    """
    names = [x for x in PKG2PROJECT if abbr in x]
    if len(names) == 1:
        return names[0]
    elif abbr in names or is_installed(abbr):
        return abbr
    return ''