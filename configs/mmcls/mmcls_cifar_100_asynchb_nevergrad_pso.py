_base_ = [
    '../_base_/context/train.py', '../_base_/searcher/nevergrad_pso.py',
    '../_base_/scheduler/asynchb.py', '../_base_/space/mmcls_model.py',
    '../_base_/space/optimizer.py', '../_base_/space/batch_size.py'
]

space = {
    'data.samples_per_gpu': {{_base_.batch_size}},
    'model': {{_base_.model}},
    'model.head.num_classes': 100,
    'optimizer': {{_base_.optimizer}},
}

task = dict(type='MMClassification')
tune_cfg = dict(num_samples=8, metric='val/accuracy_top-1', mode='max')
