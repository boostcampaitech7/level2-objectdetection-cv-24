_base_ = '../_base_/default_runtime.py'

# ========================Frequently modified parameters======================
# -----data related-----
data_root = './dataset/'
dataset_type = 'mmyolo.YOLOv5CocoDataset'
num_classes = 10
classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
           'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')

# Batch size
train_batch_size_per_gpu = 4
train_num_workers = 2

# -----model related-----
# Scaling factor
deepen_factor = 1.0
widen_factor = 1.0

# -----train val related-----
base_lr = 5e-5
max_epochs = 150

loss_cls = 0.05
loss_bbox = 0.3
loss_dfl = 0.05

momentum = 0.03
eps = 0.001

img_scale = (1024, 1024)

swin_backbone = dict(
    type='SwinTransformer',
    embed_dims=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=7,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.2,
    patch_norm=True,
    out_indices=(0, 1, 2, 3),
    with_cp=False,
    convert_weights=True,
    init_cfg=dict(type='Pretrained', 
                  checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth'))

model = dict(
    type='mmyolo.YOLODetector',
    data_preprocessor=dict(
        type='mmyolo.YOLOv5DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[255., 255., 255.],
        bgr_to_rgb=True
    ),
    backbone=swin_backbone,
    neck=dict(
        type='mmyolo.YOLOv8PAFPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=[128, 256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=momentum, eps=eps),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='mmyolo.YOLOv8Head',
        head_module=dict(
            type='mmyolo.YOLOv8HeadModule',
            num_classes=10,
            in_channels=[128, 256, 512, 1024],
            widen_factor=widen_factor,
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=momentum, eps=eps),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=[4, 8, 16, 32]),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=[4, 8, 16, 32]),
        bbox_coder=dict(type='mmyolo.DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls),
        loss_bbox=dict(
            type='mmyolo.IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=loss_dfl)),
    train_cfg=dict(
        assigner=dict(
            type='mmyolo.BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=10,
            alpha=0.5,
            beta=6.0,
            eps=1e-9)),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300,
        featuremap_strieds=[4, 8, 16, 32]))

# ========================Dataset config=========================
train_pipeline = [
    dict(type='mmyolo.LoadImageFromFile', backend_args=None),
    dict(type='mmyolo.LoadAnnotations', with_bbox=True),
    dict(
        type='mmyolo.Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        prob=0.5,
        pre_transform=[
            dict(type='mmyolo.LoadImageFromFile', backend_args=None),
            dict(type='mmyolo.LoadAnnotations', with_bbox=True)
        ]),
    dict(
        type='mmyolo.YOLOv5RandomAffine',
        max_rotate_degree=10.0,
        max_shear_degree=2.0,
        scaling_ratio_range=(0.1, 2.0),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    dict(type='mmyolo.YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='temp_train.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=dict(classes=classes)),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=4,
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
        pipeline=[
            dict(type='mmyolo.LoadImageFromFile', backend_args=None),
            dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
            dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
            dict(type='mmyolo.LoadAnnotations', with_bbox=True),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
        ],
        metainfo=dict(classes=classes)))

test_dataloader = val_dataloader

# ========================Evaluation config=======================
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

# ========================Training config=======================
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.005),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=4
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs - 1,
        eta_min=base_lr * 0.01,  # 0.05에서 0.01로 감소
        begin=1,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3,
        save_best='auto', 
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=50)
)

custom_hooks = [
    dict(
        type='mmyolo.EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP_50',  #판단 척도, bbox_mAP_50이 리더보드 점수 / 일반적인 성능이 bbox_mAP라 알아서 골라야함
        min_delta=0.001,      # 최소 향상 정도 : 이정도는 올라가야 성능 좋아진거라 봄
        patience=5,           #n에폭 연속 못올라가면 사망(위 param scheduler patience보다는 커야지 안그러면 lr조정도 안하고 early stop)
        rule='greater'        
    )
]

log_processor = dict(by_epoch=True)
    