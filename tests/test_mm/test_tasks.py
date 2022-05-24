import argparse
from unittest.mock import patch

from mmtune.mm.tasks import MMDetection, MMSegmentation, Sphere


@patch.object(MMSegmentation, 'train_model', return_value=None)
@patch.object(MMSegmentation, 'build_dataset')
def test_mmseg(mock_build_dataset, mock_train_model):
    mock_build_dataset.return_value.CLASSES = ['a', 'b', 'c']

    parser = argparse.ArgumentParser()
    config_path = 'configs/mmseg/pspnet/pspnet_r18-d8_4x4_512x512_80k_potsdam.py'  # noqa
    parser.add_argument('--config')

    task = MMSegmentation()
    parser = task.add_arguments(parser)
    args = parser.parse_args(['--config', config_path])

    task.run(args=args)


@patch.object(MMDetection, 'train_model', return_value=None)
@patch.object(MMDetection, 'build_dataset')
def test_mmdet(mock_build_dataset, mock_train_model):
    mock_build_dataset.return_value.CLASSES = ['a', 'b', 'c']

    parser = argparse.ArgumentParser()
    config_path = 'configs/mmdet/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    parser.add_argument('--config')

    task = MMDetection()
    parser = task.add_arguments(parser)
    args = parser.parse_args(['--config', config_path])

    task.run(args=args)


def test_sphere():
    parser = argparse.ArgumentParser()
    config_path = 'configs/mmtune/bbo_sphere_nevergrad_oneplusone.py'
    parser.add_argument('--config')

    task = Sphere()
    parser = task.add_arguments(parser)
    args = parser.parse_args(['--config', config_path])

    task.run(args=args)
