# Refer to https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers/median_stopping_rule.py  # noqa

trial_scheduler = dict(
    type='MedianStoppingRule',
    time_attr='time_total_s',
    grace_period=60,
    min_samples_required=3,
    min_time_slice=0,
    hard_stop=True)
