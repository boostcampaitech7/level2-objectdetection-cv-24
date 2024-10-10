_base_ = [
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

dataset_type = 'CocoDataset'
data_root = './dataset/'
classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
           'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')

# 데이터 파이프라인 수정
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='temp_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes)
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='temp_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(classes=classes)
    )
)

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

# model settings
model = dict(
    type='YOLOv5',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[255., 255., 255.],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='YOLOv5CSPDarknet',
        deepen_factor=0.33,
        widen_factor=1.0,
        out_indices=(2, 3, 4),
        # norm_cfg=dict(type='BN', momentum=0.01, eps=0.001)
        norm_cfg=dict(type='GN', num_groups=32, eps=1e-6)),
    neck=dict(
        type='YOLOv5PAFPN',
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        conv_cfg=None,
        # norm_cfg=dict(type='BN', momentum=0.01, eps=0.001),
        norm_cfg=dict(type='GN', num_groups=32, eps=1e-6),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv5Head',
        num_classes=10,
        in_channels=[256, 512, 1024],
        widen_factor=1.0,
        head_module=dict(
            type='YOLOv5HeadModule',
            num_classes=10,
            in_channels=[256, 512, 1024],
            widen_factor=1.0
        ),
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(75.1, 69.5), (162.4, 166.2), (346.4, 189.9)],
                        [(230.9, 347.1), (281.1, 579.1), (478.9, 372.4)],
                        [(749.3, 418.9), (516.3, 691.9), (874.5, 810.1)]],
            strides=[8, 16, 32]),
        bbox_coder=dict(type='YOLOv5BBoxCoder'),
        featmap_strides=[8, 16, 32],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=0.0005),
        loss_bbox=dict(
            type='YOLOIoULoss',
            iou_mode='ciou',
            eps=1e-16,
            reduction='sum',
            loss_weight=0.002),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=0.0005)),
    # model training and testing settings
    train_cfg=dict(
        _delete_=True,
        type='EpochBasedTrainLoop',
        max_epochs=40,
        val_interval=1),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100,
        multi_label=True)
    )

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    # optimizer=dict(type='Adam', lr=1e-6, betas=(0.9, 0.999)),
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    accumulative_counts=4
)

val_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    ann_file=data_root + 'temp_val.json',
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

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='StepLR',
        step_size=20,          # 20 에포크마다 학습률 감소
        gamma=0.1
    )
]
work_dir = './work_dirs/yolov5'

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50)
)

log_processor = dict(by_epoch=True)