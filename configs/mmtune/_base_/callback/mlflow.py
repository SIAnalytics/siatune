callbacks = [
    dict(
        type='MLflowLoggerCallback',
        experiment_name='siatune',
        save_artifact=True,
        metric='train/loss',
        mode='max',
    ),
]
