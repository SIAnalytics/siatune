callbacks = [
    dict(
        type='MLflowLoggerCallback',
        experiment_name='mmtune',
        save_artifact=True,
        metric='train/loss',
        mode='max',
    ),
]
