context = dict(rewriters=[
    dict(type='BatchConfigPathcer'),
    dict(type='SequeunceConfigPathcer'),
    dict(type='Decouple', keys=['searched_cfg', 'base_cfg']),
    dict(type='ConfigMerger'),
    dict(type='Dump'),
    dict(type='SetEnv')
])
