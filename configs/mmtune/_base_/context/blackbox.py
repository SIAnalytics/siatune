rewriters = [
    dict(type='BuildBaseCfg', dst_key='base_cfg'),
    dict(type='Decouple', key='searched_cfg'),
    dict(
        type='ConfigMerger',
        src_key='searched_cfg',
        dst_key='base_cfg',
        ctx_key='cfg'),
    dict(type='Dump', ctx_key='cfg', arg_key='config'),
    dict(type='BuildBaseCfg', arg_key='config', dst_key='base_cfg'),
]
