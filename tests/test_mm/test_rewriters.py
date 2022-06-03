import argparse
from unittest.mock import MagicMock, patch

import mmcv

from mmtune.mm.context.rewriters import (BatchConfigPatcher, ConfigMerger,
                                         CustomHookRegister, Decouple, Dump,
                                         InstantiateCfg, PathJoinTrialId,
                                         SequeunceConfigPatcher)


def test_build_base_cfg():
    build_base_cfg = InstantiateCfg(dst_key='base_cfg')
    context = dict()
    build_base_cfg(context)


def test_decouple():
    decouple = Decouple(key='searched_cfg')

    context = dict(
        searched_cfg=dict(model=[
            dict(type='DummyModel'),
            dict(type='DummyModel2'),
        ]), )
    decouple(context)


def test_dump():
    dump = Dump(ctx_key='cfg', arg_key='config')
    config = mmcv.Config(dict())
    args = MagicMock()
    args.config = config
    context = dict(cfg=config, args=args)
    dump(context)


@patch('ray.tune.get_trial_id')
def test_suffix_trial_id(mock_get_trial_id):
    mock_get_trial_id.return_value = '123'
    args = argparse.Namespace()
    args.work_dir = '/tmp'
    context = dict(args=args)
    suffix = PathJoinTrialId(key='work_dir')
    context = suffix(context)
    assert context['args'].work_dir == '/tmp/123'


def test_merge():
    merger = ConfigMerger(
        src_key='searched_cfg', dst_key='base_cfg', ctx_key='cfg')

    context = dict(
        base_cfg=mmcv.Config(dict(model=dict(type='DummyModel'))),
        searched_cfg=mmcv.Config(
            dict(model=[
                dict(type='DummyModel6'),
                dict(type='DummyModel2'),
            ])))
    merger(context)


def test_patch():
    context = dict(
        base_cfg=mmcv.Config(dict(model=dict(type='DummyModel'))),
        searched_cfg=mmcv.Config(
            dict(model=[
                dict(type='DummyModel6'),
                dict(type='DummyModel2'),
            ])))
    patcher = BatchConfigPatcher(key='searched_cfg')
    patcher(context)

    patcher = SequeunceConfigPatcher(key='searched_cfg')
    patcher(context)


def test_register():
    post_custom_hooks = ['a', 'b']
    register = CustomHookRegister(
        ctx_key='cfg', post_custom_hooks=post_custom_hooks)
    cfg = MagicMock()
    cfg.custom_hooks = []
    context = dict(cfg=cfg)
    context = register(context)
