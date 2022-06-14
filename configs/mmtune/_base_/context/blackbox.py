task = dict(
    rewriters=[
        dict(type='InstantiateCfg', key='base_cfg'),
        dict(
            type='ConfigMerger',
            src_key='searched_cfg',
            dst_key='base_cfg',
            key='cfg'),
        dict(type='Dump', key='cfg', arg_name='config'),
    ], )
