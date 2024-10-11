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
        type='mmdet.DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[255., 255., 255.],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmyolo.YOLOv5CSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.5,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth',
            prefix='backbone.')
    ),
    neck=dict(
        type='mmyolo.YOLOv5PAFPN',
        deepen_factor=0.33,
        widen_factor=0.5,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='mmyolo.YOLOv5Head',
        head_module=dict(
            type='mmyolo.YOLOv5HeadModule',
            num_classes=len(classes),
            in_channels=[256, 512, 1024],
            widen_factor=0.5,
            featmap_strides=[8, 16, 32],
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=[[(10, 13), (16, 30), (33, 23)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(116, 90), (156, 198), (373, 326)]],
            strides=[8, 16, 32]),
        bbox_coder=dict(
            type='mmyolo.YOLOv5BBoxCoder',
            decode_clip_border=True,
        ),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.5),
        loss_bbox=dict(
            type='mmyolo.IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-7,
            reduction='mean',
            loss_weight=0.03,
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
        scaling_ratio_range=(0.1, 1.9),
        border=(-320, -320),
        border_val=(114, 114, 114)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
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

train_dataloader = dict(
    batch_size=4,
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
test_dataloader = val_dataloader

# test evaluator
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=5e-5,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=16),
    constructor='mmyolo.YOLOv5OptimizerConstructor',
    clip_grad=dict(max_norm=10, norm_type=2))

# learning rate (LinearLR, MultiStepLR)
# param_scheduler = [
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=300,
#         by_epoch=True,
#         milestones=[250],
#         gamma=0.1)
# ]

# # CosineAnnealingLR
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=300,  # 총 에폭 수
        eta_min=1e-5,  # 최소 학습률
        begin=0,
        end=300,
        by_epoch=True,
        convert_to_iter_based=True)
]

# ReduceLROnPlateau
# param_scheduler = [
#     dict(
#         type='ReduceLROnPlateau',
#         mode='min',
#         factor=0.1,
#         patience=10,
#         verbose=True,
#         threshold=0.0001,
#         threshold_mode='rel',
#         cooldown=0,
#         min_lr=0,
#         eps=1e-8
#     )
# ]

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