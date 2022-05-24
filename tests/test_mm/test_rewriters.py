from unittest.mock import MagicMock, patch

import mmcv

from mmtune.mm.context.rewriters import (BatchConfigPathcer, ConfigMerger,
                                         CustomHookRegister, Decouple, Dump,
                                         SequeunceConfigPathcer, SetEnv)


def test_decouple():
    decouple = Decouple(keys=['searched_cfg', 'base_cfg'])

    context = dict(
        base_cfg=dict(model=dict(type='DummyModel')),
        searched_cfg=dict(model=[
            dict(type='DummyModel'),
            dict(type='DummyModel2'),
        ]),
    )
    decouple(context)


def test_dump():
    dump = Dump()
    config = mmcv.Config(dict())
    args = MagicMock()
    args.config = config
    context = dict(cfg=config, args=args)
    dump(context)


@patch('ray.tune.get_trial_id')
def test_setenv(mock_get_trial_id):
    mock_get_trial_id.return_value = 'sdfkj234'
    setenv = SetEnv()

    args = MagicMock()
    args.work_dir = 'tmpdir'
    context = dict(args=args)
    setenv(context)


def test_merge():
    merger = ConfigMerger()

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
    patcher = BatchConfigPathcer()
    patcher(context)

    patcher = SequeunceConfigPathcer()
    patcher(context)


def test_register():
    post_custom_hooks = ['a', 'b']
    register = CustomHookRegister(post_custom_hooks)
    cfg = MagicMock()
    cfg.custom_hooks = []
    context = dict(cfg=cfg)
    context = register(context)
