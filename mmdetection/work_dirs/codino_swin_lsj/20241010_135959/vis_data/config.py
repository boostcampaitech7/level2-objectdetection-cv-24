auto_scale_lr = dict(base_batch_size=16)
backend_args = None
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
        'projects.AlignDETR.align_detr',
    ])
data_root = '../dataset/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, type='CheckpointHook'),
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
max_epochs = 12
model = dict(
    as_two_stage=True,
    backbone=dict(
        depth=50,
        frozen_stages=0,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='FrozenBN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        all_layers_num_gt_repeat=[
            2,
            2,
            2,
            2,
            2,
            1,
            2,
        ],
        alpha=0.25,
        gamma=2.0,
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=80,
        sync_cls_avg_factor=True,
        tau=1.5,
        type='AlignDETRHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=1,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_layers=6),
    neck=dict(
        act_cfg=None,
        bias=True,
        in_channels=[
            512,
            1024,
            2048,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=-0.5, temperature=10000),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='MixedHungarianAssigner')),
    type='DINO',
    with_box_refine=True)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.0001),
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
    batch_size=2,
    dataset=dict(
        ann_file='train.json',
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
    ann_file='../dataset/test.json',
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
work_dir = './mmdetection/work_dirs/codino_swin_lsj'
