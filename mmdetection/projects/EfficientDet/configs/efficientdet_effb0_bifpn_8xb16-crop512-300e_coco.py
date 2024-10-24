_base_ = 'mmdet::_base_/default_runtime.py'

custom_imports = dict(
    imports=['projects.EfficientDet.efficientdet'], allow_failed_imports=False)

image_size = 512
batch_augments = [
    dict(type='BatchFixedSizePad', size=(image_size, image_size))
]
# dataset settings
dataset_type = 'CocoDataset'
data_root = '../dataset/'
classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
           'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')

evalute_type = 'CocoMetric'
norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32-aa-advprop_in1k_20220119-26434485.pth'  # noqa
model = dict(
    type='EfficientDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=image_size,
        batch_augments=batch_augments),
    backbone=dict(
        type='EfficientNet',
        arch='b0',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        conv_cfg=dict(type='Conv2dSamePadding'),
        norm_cfg=norm_cfg,
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)),
    neck=dict(
        type='BiFPN',
        num_stages=3,
        in_channels=[40, 112, 320],
        out_channels=64,
        start_level=0,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='EfficientDetSepBNHead',
        num_classes=10, # 클래스 개수 수정
        num_ins=5,
        in_channels=64,
        feat_channels=64,
        stacked_convs=3,
        norm_cfg=norm_cfg,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.5,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='HuberLoss', beta=0.1, loss_weight=50)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(
            type='PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(
            type='soft_nms',
            iou_threshold=0.3,
            sigma=0.5,
            min_score=1e-3,
            method='gaussian'),
        max_per_img=100))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=(1024, 1024),
        pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(image_size, image_size),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(image_size, image_size)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='temp_train.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            metainfo=dict(classes=classes)),
        pipeline=train_pipeline
    ))

# validation dataloader
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='temp_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(classes=classes)))

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=classes)
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'temp_val.json',
    metric='bbox',
    format_only=False,
    classwise=True
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric='bbox',
    format_only=False,
    classwise=True
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-5, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)

# learning policy
max_epochs = 300
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=3),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,
        begin=3,
        T_max=297,
        end=300,
        by_epoch=True,
        convert_to_iter_based=True)
]
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=300,
    val_interval=5
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]
# cudnn_benchmark=True can accelerate fix-size training
env_cfg = dict(cudnn_benchmark=True)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (16 samples per GPU)
auto_scale_lr = dict(base_batch_size=128)
