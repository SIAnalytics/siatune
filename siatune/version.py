# Copyright (c) SI-Analytics. All rights reserved.
__version__ = '0.2.0'
import mmcv
from mmengine.utils import digit_version

IS_DEPRECATED_MMCV = False

if (digit_version(mmcv.__version__) < digit_version('2.0.0rc0')):
    IS_DEPRECATED_MMCV = True

version_info = digit_version(__version__)
