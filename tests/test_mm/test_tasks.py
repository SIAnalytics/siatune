from unittest.mock import MagicMock, patch

from mmtune.mm.tasks import (MMClassification, MMDetection, MMSegmentation,
                             Sphere)


@patch.object(MMSegmentation, 'train_model', return_value=None)
@patch.object(MMSegmentation, 'build_dataset')
def test_mmseg(mock_build_dataset, mock_train_model):
    mock_build_dataset.return_value.CLASSES = ['a', 'b', 'c']

    config_path = 'configs/mmseg/pspnet/pspnet_r18-d8_4x4_512x512_80k_potsdam.py'  # noqa

    task = MMSegmentation()
    task.set_args([config_path])
    args = MagicMock()
    args.config = config_path
    task.run(args=args)


@patch.object(MMDetection, 'train_model', return_value=None)
@patch.object(MMDetection, 'build_dataset')
def test_mmdet(mock_build_dataset, mock_train_model):
    mock_build_dataset.return_value.CLASSES = ['a', 'b', 'c']

    config_path = 'configs/mmdet/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

    task = MMDetection()
    task.set_args([config_path])
    args = MagicMock()
    args.config = config_path
    task.run(args=args)


@patch.object(MMClassification, 'train_model', return_value=None)
@patch.object(MMClassification, 'build_dataset')
def test_mmcls(mock_build_dataset, mock_train_model):
    mock_build_dataset.return_value.CLASSES = ['a', 'b', 'c']

    config_path = 'configs/mmcls/resnet/resnet18_8xb16_cifar10.py'

    task = MMClassification()
    task.set_args([config_path])
    args = MagicMock()
    args.config = config_path
    task.run(args=args)


def test_sphere():
    # config_path = 'configs/mmtune/bbo_sphere_nevergrad_oneplusone.py'

    task = Sphere()  # noqa
    # TODO: in progress
    # task.set_args([config_path])
    # task.run()
