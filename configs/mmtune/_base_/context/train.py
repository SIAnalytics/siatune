post_custom_hooks = [
    dict(type='RayTuneLoggerHook', filtering_key='val', priority='VERY_LOW'),
    dict(type='RayCheckpointHook', by_epoch=True, interval=1)
]

task = dict(rewriters=[
    dict(type='BuildBaseCfg', arg_key='config', dst_key='base_cfg'),
    dict(type='BatchConfigPathcer', key='searched_cfg'),
    dict(type='SequeunceConfigPathcer', key='searched_cfg'),
    dict(type='Decouple', key='searched_cfg'),
    dict(
        type='ConfigMerger',
        src_key='searched_cfg',
        dst_key='base_cfg',
        ctx_key='cfg'),
    dict(
        type='CustomHookRegister',
        ctx_key='cfg',
        post_custom_hooks=post_custom_hooks),
    dict(type='Dump', ctx_key='cfg', arg_key='config'),
    dict(type='SuffixTrialId', key='work_dir')
])
