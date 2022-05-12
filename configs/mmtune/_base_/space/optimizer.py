optimizer = dict(
    type='Choice',
    categories=[
        dict(
            type='SGD',
            lr=0.01,
            _delete_=True,
        ),
        dict(
            type='RMSprop',
            _delete_=True,
        ),
        dict(
            type='Adam',
            _delete_=True,
        ),
        dict(
            type='AdamW',
            _delete_=True,
        ),
    ],
    alias=['sgd', 'rms', 'adam', 'adamw'],
)
