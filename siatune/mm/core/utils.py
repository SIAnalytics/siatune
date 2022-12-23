# Copyright (c) SI-Analytics. All rights reserved.

import mmcv
from mmengine.utils import digit_version

IS_DEPRECATED_MMCV = False

if (digit_version(mmcv.__version__) < digit_version('2.0.0')):
    IS_DEPRECATED_MMCV = True
