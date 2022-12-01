sync_norm_cfg = dict(type='SyncBN', requires_grad=True)
# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))

resnet18 = dict(
    _delete_=True,
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

convnext_t = dict(
    _delete_=True,
    type='ImageClassifier',
    backbone=dict(
        type='ConvNeXt',
        arch='tiny',
        out_indices=(3, ),
        drop_path_rate=0.1,
        gap_before_final_norm=True,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
        ]),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

model = dict(
    type='Choice',
    categories=[
        resnet18,
        convnext_t,
    ],
    alias=[
        'resnet18',
        'convnext_t',
    ],
)
