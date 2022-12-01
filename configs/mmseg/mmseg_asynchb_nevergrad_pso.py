_base_ = [
    '../_base_/context/train.py', '../_base_/searcher/nevergrad_pso.py',
    '../_base_/scheduler/asynchb.py', '../_base_/space/mmseg_model.py',
    '../_base_/space/optimizer.py', '../_base_/space/batch_size.py'
]

space = dict(
    data=dict(samples_per_gpu={{_base_.batch_size}}),
    model={{_base_.model}},
    optimizer={{_base_.optimizer}})

task = dict(type='MMSegmentation')
tune_cfg = dict(num_samples=8, metric='val/mIoU', mode='max')
