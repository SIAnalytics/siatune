_base_ = [
    '../_base_/context/train_rargs.py', '../_base_/searcher/nevergrad_pso.py',
    '../_base_/scheduler/asynchb.py', '../_base_/space/mmcls_model.py',
    '../_base_/space/optimizer.py', '../_base_/space/batch_size.py'
]

space = {
    'model': {{_base_.model}},
    'model.head.num_classes': 100,
    'optimizer': {{_base_.optimizer}},
    'data.samples_per_gpu': {{_base_.batch_size}},
}

task = dict(type='MIM', pkg_name='mmcls')
tune_cfg = dict(
    num_samples=8, metric='val/accuracy_top-1', mode='max', reuse_actors=False)
