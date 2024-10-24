_base_ = '../_base_/default_runtime.py'

# model settings
custom_imports = dict(imports=['mmyolo.models'], allow_failed_imports=False)

# dataset settings
dataset_type = 'CocoDataset'
data_root = './dataset/'
classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
           'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')

model = dict(
    type='mmyolo.YOLODetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[255., 255., 255.],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmyolo.YOLOv5CSPDarknet',
        deepen_factor=0.67,  # YOLOv5m에 맞게 수정
        widen_factor=0.75,   # YOLOv5m에 맞게 수정
        norm_cfg=dict(type='BN', momentum=0.01, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944-516a710f.pth',  # YOLOv5m 가중치 파일 경로
            prefix='backbone.')
    ),
    neck=dict(
        type='mmyolo.YOLOv5PAFPN',
        deepen_factor=0.67,  # YOLOv5m에 맞게 수정
        widen_factor=0.75,   # YOLOv5m에 맞게 수정
        in_channels=[256, 512, 1024],  # YOLOv5m에 맞게 수정
        out_channels=[256, 512, 1024],  # YOLOv5m에 맞게 수정
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.01, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='mmyolo.YOLOv5Head',
        head_module=dict(
            type='mmyolo.YOLOv5HeadModule',
            num_classes=len(classes),
            in_channels=[256, 512, 1024],  # YOLOv5m에 맞게 수정
            widen_factor=0.75,  # YOLOv5m에 맞게 수정
            featmap_strides=[8, 16, 32],
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=[
                [(68.97, 58.18), (112.36, 145.61), (235.20, 132.57)],
                [(185.42, 281.96), (390.79, 245.50), (296.98, 447.08)],
                [(625.19, 386.70), (404.13, 683.98), (821.21, 720.49)],
            ],
            strides=[8, 16, 32]
        ),
        bbox_coder=dict(
            type='mmyolo.YOLOv5BBoxCoder',
            decode_clip_border=True,
        ),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=0.1),
        loss_bbox=dict(
            type='mmyolo.IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-7,
            reduction='mean',
            loss_weight=0.1,
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.5),
        prior_match_thr=4.0,
        obj_level_weights=[4.0, 1.0, 0.4]),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=300))

# training settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=(1024, 1024),
        pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-512, -512)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
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
    )
)

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
    optimizer=dict(type='AdamW', lr=2e-5, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)

param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001, 
        by_epoch=True, 
        begin=0, 
        end=3  # 약 3 에폭에 해당하는 warmup
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=297,  # 300 에폭 - warmup 3 에폭
        eta_min=1e-6,
        begin=3,
        end=300,
        by_epoch=True,
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