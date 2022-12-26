_base_ = [
    '../_base_/context/irresponsible_train.py',
    '../_base_/searcher/nevergrad_pso.py', '../_base_/scheduler/asynchb.py',
    '../_base_/space/mmcls_model.py', '../_base_/space/optimizer.py',
    '../_base_/space/batch_size.py'
]

space = {
    'model': {{_base_.model}},
    'model.head.num_classes': 100,
    'optimizer': {{_base_.optimizer}},
    'data.samples_per_gpu': {{_base_.batch_size}},
}

metric = 'accuracy_top-1'
task = dict(type='MMAny', pkg_name='mmcls')
tune_cfg = dict(
    num_samples=8, metric='accuracy_top-1', mode='max', reuse_actors=False)
