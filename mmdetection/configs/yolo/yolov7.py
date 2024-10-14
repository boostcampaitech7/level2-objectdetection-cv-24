_base_ = '../_base_/default_runtime.py'

# model settings
custom_imports = dict(imports=['mmyolo.models', 'mmyolo.utils'], allow_failed_imports=False)

# dataset settings
dataset_type = 'CocoDataset'
data_root = './dataset/'
classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
           'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')

model = dict(
    type='mmyolo.YOLODetector',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[255., 255., 255.],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmyolo.YOLOv7Backbone',
        arch='W',  # 또는 'E' for YOLOv7-E
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    neck=dict(
        type='mmyolo.YOLOv7PAFPN',
        in_channels=[512, 1024, 1024],  # YOLOv7-W에 맞게 수정
        out_channels=[128, 256, 512],
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='mmyolo.YOLOv7Head',
        head_module=dict(
            type='mmyolo.YOLOv7HeadModule',
            num_classes=len(classes),
            in_channels=[128, 256, 512],  # YOLOv7-W에 맞게 수정
            featmap_strides=[8, 16, 32],
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=[
                [(72.18, 67.92), (158.48, 156.68), (348.69, 182.71)],
                [(219.74, 321.61), (263.29, 550.62), (454.32, 372.42)],
                [(725.55, 403.40), (489.98, 687.33), (863.31, 787.60)],
            ],
            strides=[8, 16, 32]
        ),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.3),
        loss_bbox=dict(
            type='mmyolo.IOU2DLoss',
            mode='ciou',
            eps=1e-7,
            reduction='mean',
            loss_weight=0.05,
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.7),
        prior_match_thr=4.0,
        obj_level_weights=[4.0, 1.0, 0.4]),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))

# training settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmyolo.YOLOv5RandomAffine',
        max_rotate_degree=0,
        max_shear_degree=0,
        scaling_ratio_range=(0.1, 2.0),
        border=(-320, -320),
        border_val=(114, 114, 114)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.Resize',
        scale=(1024, 1024),
        keep_ratio=True,
        clip_object_border=False),
    dict(
        type='mmdet.Pad',
        size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

# training settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmyolo.YOLOv5RandomAffine',
        max_rotate_degree=20,
        max_translate_ratio=0.2,
        scaling_ratio_range=(0.1, 2.0),
        max_shear_degree=5,
        border=(-320, -320),
        border_val=(114, 114, 114)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.Resize',
        scale=(1024, 1024),
        keep_ratio=True,
        clip_object_border=False),
    dict(
        type='mmdet.Pad',
        size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='temp_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes)))

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

# validation evaluator
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'temp_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    proposal_nums=(100, 1, 10),
    iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    metric_items=['mAP', 'mAP_50', 'mAP_75']
)

# test dataloader
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

# test evaluator
test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'test.json',
    metric='bbox',
    format_only=False,
    classwise=True
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0005),
    clip_grad=dict(max_norm=10.0, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001, 
        by_epoch=False, 
        begin=0, 
        end=1000
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=300,
        eta_min=1e-6,
        begin=1000,
        end=300000,
        by_epoch=False,
        convert_to_iter_based=True
    )
]

# runtime settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=300,
    val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))