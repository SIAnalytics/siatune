_base_ = [
    '../_base_/context/train.py', '../_base_/searcher/nevergrad_pso.py',
    '../_base_/scheduler/asynchb.py', 'configs/_base_/space/mmseg_model.py',
    'configs/_base_/space/optimizer.py'
]

metric = 'val/mIoU'
mode = 'max'
raise_on_failed_trial = False,
num_samples = 256
