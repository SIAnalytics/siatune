data = dict(workers_per_gpu=1, train=dict())
model = dict()
train_cfg = None
test_cfg = None
checkpoint_config = dict(interval=-1)
log_level = 'INFO'
work_dir = ''
workflow = [('train', 1)]
