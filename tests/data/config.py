data = dict(workers_per_gpu=1, train=dict())
model = dict()
checkpoint_config = dict(interval=-1)
log_level = 'INFO'
workflow = [('train', 1)]
