_base_ = [
    './_base_/context/train.py', './_base_/searcher/nevergrad_pso.py',
    './_base_/scheduler/asynchb.py', './_base_/space/mmdet_model.py',
    './_base_/space/optimizer.py', './_base_/space/batch_size.py'
]

space = {
    'model': {{_base_.model}},
    'optimizer': {{_base_.optimizer}},
    'data.samples_per_gpu': {{_base_.batch_size}},
}

task = 'MMDetection'
metric = 'val/AP'
mode = 'max'
raise_on_failed_trial = False
num_samples = 256
