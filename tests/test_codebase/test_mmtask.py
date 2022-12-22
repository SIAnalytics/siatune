from unittest.mock import patch

import mmcls  # noqa: F401
import mmdet  # noqa: F401
import mmedit  # noqa: F401
import mmseg  # noqa: F401

from siatune.codebase import (MMClassification, MMDetection, MMEditing,
                              MMSegmentation)


@patch('mmcls.apis.train_model')
@patch('mmcls.datasets.build_dataset')
@patch('mmcls.models.build_classifier')
def test_mmcls(*mocks):
    task = MMClassification(args=['tests/data/config.py'], num_workers=1)
    task.run(args=task.args)


@patch('mmdet.apis.train_detector')
@patch('mmdet.datasets.build_dataset')
@patch('mmdet.models.build_detector')
def test_mmdet(*mocks):
    task = MMDetection(args=['tests/data/config.py'], num_workers=1)
    task.run(args=task.args)


@patch('mmedit.apis.train_model')
@patch('mmedit.datasets.build_dataset')
@patch('mmedit.models.build_model')
def test_mmedit(*mocks):
    task = MMEditing(args=['tests/data/config.py'], num_workers=1)
    task.run(args=task.args)


@patch('mmseg.apis.train_segmentor')
@patch('mmseg.datasets.build_dataset')
@patch('mmseg.models.build_segmentor')
def test_mmseg(*mocks):
    task = MMSegmentation(args=['tests/data/config.py'], num_workers=1)
    task.run(args=task.args)
