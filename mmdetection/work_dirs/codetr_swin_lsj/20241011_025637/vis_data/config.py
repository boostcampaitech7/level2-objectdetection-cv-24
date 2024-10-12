auto_scale_lr = dict(base_batch_size=16)
backend_args = None
batch_augments = [
    dict(pad_mask=True, size=(
        1024,
        1024,
    ), type='BatchFixedSizePad'),
]
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
    allow_failed_imports=True, imports=[
        'projects.CO-DETR.codetr',
    ])
data_root = '../dataset/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True, interval=1, max_keep_ckpts=3, type='CheckpointHook'),
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
image_size = (
    1024,
    1024,
)
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_lambda = 2.0
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=[
        dict(
            anchor_generator=dict(
                octave_base_scale=8,
                ratios=[
                    1.0,
                ],
                scales_per_octave=1,
                strides=[
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                ],
                type='AnchorGenerator'),
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            feat_channels=256,
            in_channels=256,
            loss_bbox=dict(loss_weight=24.0, type='GIoULoss'),
            loss_centerness=dict(
                loss_weight=12.0, type='CrossEntropyLoss', use_sigmoid=True),
            loss_cls=dict(
                alpha=0.25,
                gamma=2.0,
                loss_weight=12.0,
                type='FocalLoss',
                use_sigmoid=True),
            num_classes=10,
            stacked_convs=1,
            type='CoATSSHead'),
    ],
    data_preprocessor=dict(
        batch_augments=[
            dict(pad_mask=True, size=(
                1024,
                1024,
            ), type='BatchFixedSizePad'),
        ],
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    eval_module='detr',
    neck=dict(
        act_cfg=None,
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=5,
        out_channels=256,
        type='ChannelMapper'),
    query_head=dict(
        as_two_stage=True,
        dn_cfg=dict(
            box_noise_scale=1.0,
            group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
            label_noise_scale=0.5),
        in_channels=2048,
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=10,
        num_query=900,
        positional_encoding=dict(
            normalize=True,
            num_feats=128,
            temperature=20,
            type='SinePositionalEncoding'),
        transformer=dict(
            decoder=dict(
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            dropout=0.0,
                            embed_dims=256,
                            num_heads=8,
                            type='MultiheadAttention'),
                        dict(
                            dropout=0.0,
                            embed_dims=256,
                            num_levels=5,
                            type='MultiScaleDeformableAttention'),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'cross_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='DetrTransformerDecoderLayer'),
                type='DinoTransformerDecoder'),
            encoder=dict(
                num_layers=6,
                transformerlayers=dict(
                    attn_cfgs=dict(
                        dropout=0.0,
                        embed_dims=256,
                        num_levels=5,
                        type='MultiScaleDeformableAttention'),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='BaseTransformerLayer'),
                type='DetrTransformerEncoder',
                with_cp=6),
            num_co_heads=2,
            num_feature_levels=5,
            type='CoDinoTransformer',
            with_coord_feat=False),
        type='CoDINOHead'),
    roi_head=[
        dict(
            bbox_head=dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(loss_weight=120.0, type='GIoULoss'),
                loss_cls=dict(
                    loss_weight=12.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=10,
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
            bbox_roi_extractor=dict(
                featmap_strides=[
                    4,
                    8,
                    16,
                    32,
                    64,
                ],
                finest_scale=56,
                out_channels=256,
                roi_layer=dict(
                    output_size=7, sampling_ratio=0, type='RoIAlign'),
                type='SingleRoIExtractor'),
            type='CoStandardRoIHead'),
    ],
    rpn_head=dict(
        anchor_generator=dict(
            octave_base_scale=4,
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                4,
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=12.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=12.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=[
        dict(max_per_img=300, nms=dict(iou_threshold=0.8, type='soft_nms')),
        dict(
            rcnn=dict(
                max_per_img=100,
                nms=dict(iou_threshold=0.5, type='nms'),
                score_thr=0.0),
            rpn=dict(
                max_per_img=1000,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=1000)),
        dict(
            max_per_img=100,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.6, type='nms'),
            nms_pre=1000,
            score_thr=0.0),
    ],
    train_cfg=[
        dict(
            assigner=dict(
                match_costs=[
                    dict(type='FocalLossCost', weight=2.0),
                    dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                    dict(iou_mode='giou', type='IoUCost', weight=2.0),
                ],
                type='HungarianAssigner')),
        dict(
            rcnn=dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            rpn=dict(
                allowed_border=-1,
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                    min_pos_iou=0.3,
                    neg_iou_thr=0.3,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=False,
                    neg_pos_ub=-1,
                    num=256,
                    pos_fraction=0.5,
                    type='RandomSampler')),
            rpn_proposal=dict(
                max_per_img=1000,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=4000)),
        dict(
            allowed_border=-1,
            assigner=dict(topk=9, type='ATSSAssigner'),
            debug=False,
            pos_weight=-1),
    ],
    type='CoDETR',
    use_lsj=True)
num_classes = 10
num_dec_layer = 6
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=2e-05, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=40,
        gamma=0.1,
        milestones=[
            11,
            30,
        ],
        type='MultiStepLR'),
]
resume = False
seed = 42
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='test.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='../dataset/',
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
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='../dataset/test.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(type='PackDetInputs'),
]
train_cfg = dict(max_epochs=40, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=1,
    dataset=dict(
        ann_file='temp_train.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='../dataset/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
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
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='temp_val.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='../dataset/',
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
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='../dataset/temp_val.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs'),
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
work_dir = './mmdetection/work_dirs/codetr_swin_lsj'
