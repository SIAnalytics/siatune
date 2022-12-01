task = dict(
    rewriters=[
        dict(type='InstantiateCfg', key='base_cfg'),
        dict(
            type='MergeConfig',
            src_key='searched_cfg',
            dst_key='base_cfg',
            save_key='cfg'),
        dict(type='Dump', key='cfg', arg_name='config'),
    ], )
