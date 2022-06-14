post_custom_hooks = [
    dict(type='RayTuneLoggerHook', filtering_key='val', priority='VERY_LOW'),
    dict(type='RayCheckpointHook', by_epoch=True, interval=1)
]

task = dict(rewriters=[
    dict(type='InstantiateCfg', arg_name='config', key='base_cfg'),
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
        post_custom_hooks=post_custom_hooks),
    dict(type='Dump', key='cfg', arg_name='config'),
    dict(type='AppendTrialIDtoPath', arg_name='work_dir')
])
