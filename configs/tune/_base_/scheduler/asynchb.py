scheduler = dict(
    type='AsyncHyperBandScheduler',
    time_attr='training_iteration',
    max_t=20,
    grace_period=2)
