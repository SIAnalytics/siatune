_base_ = [
    '../_base_/context/train.py', '../_base_/searcher/nevergrad_pso.py',
    '../_base_/scheduler/asynchb.py', '../_base_/space/mmseg_model.py',
    '../_base_/space/optimizer.py', '../_base_/space/batch_size.py'
]

space = {
    'data.samples_per_gpu': {{_base_.batch_size}},
    'model': {{_base_.model}},
    'model.decode_head.num_classes': 21,
    'model.auxiliary_head.num_classes': 21,
    'optimizer': {{_base_.optimizer}},
}

task = dict(type='MIM', pkg_name='mmsegmentation')
tune_cfg = dict(
    num_samples=8,
    metric='val/mIoU',
    mode='max',
    reuse_actors=False,
    chdir_to_trial_dir=False)
