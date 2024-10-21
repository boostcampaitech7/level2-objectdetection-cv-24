_base_ = [
    '../../configs/cascade_rcnn/cascade-rcnn_x101_64x4d_fpn_20e_coco.py'
]
import sys
import os

# custom_hooks.py가 위치한 디렉토리 경로를 추가
sys.path.append(os.path.abspath('/data/ephemeral/home/level2-objectdetection-cv-24/mmdetection'))

# Custom imports 설정
custom_imports = dict(
    imports=['custom_hooks'],  # custom_hooks.py 파일을 임포트
    allow_failed_imports=True
)

dataset_type = 'CocoDataset'
data_root = './dataset/'
classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
           'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')

# 데이터 파이프라인 설정
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Albu',
        transforms=[
            dict(
            type='Sharpen',
            alpha=(0.2, 0.5),
            lightness=(0.5, 1.5), 
            p=0.5 
        )
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=True
    ),
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

# 데이터 로더 설정
train_dataloader = dict(
    batch_size=8,
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
    batch_size=4,
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

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    #paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
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

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='ReduceOnPlateauParamScheduler',
        param_name='lr',  
        monitor='coco/bbox_mAP_50',   
        rule='greater',      
        factor=0.5,       
        patience=3,      
        min_value=1e-6,   
        by_epoch=True     
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,  
        max_keep_ckpts=3,  
        save_best='bbox_mAP_50', 
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=50)
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',  #판단 척도, bbox_mAP_50이 리더보드 점수 / 일반적인 성능이 bbox_mAP라 알아서 골라야함
        min_delta=0.001,      # 최소 향상 정도 : 이정도는 올라가야 성능 좋아진거라 봄
        patience=7,           #n에폭 연속 못올라가면 사망(위 param scheduler patience보다는 커야지 안그러면 lr조정도 안하고 early stop)
        rule='greater'        
    )
]

log_processor = dict(by_epoch=True)
