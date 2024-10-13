classes = (
    'General trash',
    'Paper',
    'Paper pack',
    'Metal',
    'Glass',
    'Plastic',
    'Styrofoam',
    'Plastic bag',
    'Battery',
    'Clothing',
)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmyolo.models',
    ])
data_root = './dataset/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
device = 'cuda'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.67,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944-516a710f.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        type='mmyolo.YOLOv5CSPDarknet',
        widen_factor=0.75),
    bbox_head=dict(
        bbox_coder=dict(
            decode_clip_border=True, type='mmyolo.YOLOv5BBoxCoder'),
        head_module=dict(
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                1024,
            ],
            num_base_priors=3,
            num_classes=10,
            type='mmyolo.YOLOv5HeadModule',
            widen_factor=0.75),
        loss_bbox=dict(
            bbox_format='xywh',
            eps=1e-07,
            iou_mode='ciou',
            loss_weight=0.1,
            reduction='mean',
            return_iou=True,
            type='mmyolo.IoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=0.1,
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True),
        loss_obj=dict(
            loss_weight=0.5,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        obj_level_weights=[
            4.0,
            1.0,
            0.4,
        ],
        prior_generator=dict(
            base_sizes=[
                [
                    (
                        68.97,
                        58.18,
                    ),
                    (
                        112.36,
                        145.61,
                    ),
                    (
                        235.2,
                        132.57,
                    ),
                ],
                [
                    (
                        185.42,
                        281.96,
                    ),
                    (
                        390.79,
                        245.5,
                    ),
                    (
                        296.98,
                        447.08,
                    ),
                ],
                [
                    (
                        625.19,
                        386.7,
                    ),
                    (
                        404.13,
                        683.98,
                    ),
                    (
                        821.21,
                        720.49,
                    ),
                ],
            ],
            strides=[
                8,
                16,
                32,
            ],
            type='mmdet.YOLOAnchorGenerator'),
        prior_match_thr=4.0,
        type='mmyolo.YOLOv5Head'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0,
            0,
            0,
        ],
        pad_size_divisor=32,
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.67,
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        num_csp_blocks=3,
        out_channels=[
            256,
            512,
            1024,
        ],
        type='mmyolo.YOLOv5PAFPN',
        widen_factor=0.75),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    type='mmyolo.YOLODetector')
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=2e-05, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=3, start_factor=0.001, type='LinearLR'),
    dict(
        T_max=297,
        begin=3,
        by_epoch=True,
        convert_to_iter_based=True,
        end=300,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
resume = False
seed = 42
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='test.json',
        data_prefix=dict(img=''),
        data_root='./dataset/',
        metainfo=dict(
            classes=(
                'General trash',
                'Paper',
                'Paper pack',
                'Metal',
                'Glass',
                'Plastic',
                'Styrofoam',
                'Plastic bag',
                'Battery',
                'Clothing',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='./dataset/test.json',
    classwise=True,
    format_only=False,
    metric='bbox',
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(type='PackDetInputs'),
]
train_cfg = dict(max_epochs=300, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=16,
    dataset=dict(
        dataset=dict(
            ann_file='temp_train.json',
            data_prefix=dict(img=''),
            data_root='./dataset/',
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=dict(
                classes=(
                    'General trash',
                    'Paper',
                    'Paper pack',
                    'Metal',
                    'Glass',
                    'Plastic',
                    'Styrofoam',
                    'Plastic bag',
                    'Battery',
                    'Clothing',
                )),
            pipeline=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            type='CocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(img_scale=(
                1024,
                1024,
            ), pad_val=114.0, type='Mosaic'),
            dict(
                border=(
                    -512,
                    -512,
                ),
                scaling_ratio_range=(
                    0.1,
                    2,
                ),
                type='RandomAffine'),
            dict(type='YOLOXHSVRandomAug'),
            dict(prob=0.5, type='RandomFlip'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(
                pad_to_square=True,
                pad_val=dict(img=(
                    114.0,
                    114.0,
                    114.0,
                )),
                type='Pad'),
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type='FilterAnnotations'),
            dict(type='PackDetInputs'),
        ],
        type='MultiImageMixDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(img_scale=(
        1024,
        1024,
    ), pad_val=114.0, type='Mosaic'),
    dict(
        border=(
            -512,
            -512,
        ),
        scaling_ratio_range=(
            0.1,
            2,
        ),
        type='RandomAffine'),
    dict(type='YOLOXHSVRandomAug'),
    dict(prob=0.5, type='RandomFlip'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(
        pad_to_square=True,
        pad_val=dict(img=(
            114.0,
            114.0,
            114.0,
        )),
        type='Pad'),
    dict(keep_empty=False, min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='temp_val.json',
        data_prefix=dict(img=''),
        data_root='./dataset/',
        metainfo=dict(
            classes=(
                'General trash',
                'Paper',
                'Paper pack',
                'Metal',
                'Glass',
                'Plastic',
                'Styrofoam',
                'Plastic bag',
                'Battery',
                'Clothing',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    1024,
                    1024,
                ),
                type='mmdet.Pad'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='./dataset/temp_val.json',
    classwise=True,
    format_only=False,
    iou_thrs=[
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ],
    metric='bbox',
    metric_items=[
        'mAP',
        'mAP_50',
        'mAP_75',
    ],
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            1024,
            1024,
        ),
        type='mmdet.Pad'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './mmdetection/work_dirs/yolov5'
