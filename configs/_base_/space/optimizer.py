space = dict(
    optimizer=dict(
        type='Choice',
        categories=[
            dict(type='SGD'),
            dict(type='RMSprop'),
            dict(type='Adam'),
            dict(type='AdamW'),
        ],
        alias=['sgd', 'rms', 'adam', 'adamw'],
    ))
