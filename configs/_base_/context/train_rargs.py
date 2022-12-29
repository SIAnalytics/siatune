task = dict(rewriters=[
    dict(type='RawArgInstantiateCfg', key='base_cfg'),
    dict(type='BatchConfigPatcher', key='searched_cfg'),
    dict(type='SequeunceConfigPatcher', key='searched_cfg'),
    dict(
        type='MergeConfig',
        src_key='searched_cfg',
        dst_key='base_cfg',
        key='cfg'),
    dict(
        type='CustomHookRegister',
        key='cfg',
        post_custom_hooks=[
            dict(
                type='RayTuneLoggerHook',
                filtering_key='val',
                priority='VERY_LOW'),
        ]),
    dict(type='RawArgResumeFromCkpt'),
    dict(type='RawArgDump', key='cfg'),
    dict(type='RawArgAppendTrialIDtoPath')
])
