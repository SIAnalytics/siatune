_base_ = [
    './_base_/context/train.py', './_base_/searcher/nevergrad_pso.py',
    './_base_/scheduler/asynchb.py', './_base_/space/mmcls_model.py',
    './_base_/space/optimizer.py', './_base_/space/batch_size.py'
]

space = {
    'model': {{_base_.model}},
    'model.head.num_classes': dict(type='Constant', value=100),
    'optimizer': {{_base_.optimizer}},
    'data.samples_per_gpu': {{_base_.batch_size}},
}

task = dict(type='MMClassification')
metric = 'val/AP'
mode = 'max'
raise_on_failed_trial = False
num_samples = 256
