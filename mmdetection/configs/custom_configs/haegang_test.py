_base_ = [
    '../../configs/cascade_rcnn/cascade_rcnn_swinL.py'
]

dataset_type = 'CocoDataset'
data_root = '../dataset/'
classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
           'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')

# 데이터 파이프라인 설정
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
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
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
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
    max_epochs=40,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='ReduceOnPlateauParamScheduler',
        param_name='lr',  
        monitor='coco/bbox_mAP',   
        rule='greater',      
        factor=0.1,       
        patience=5,      
        min_value=1e-5,   
        by_epoch=True     
    )
]

work_dir = './work_dirs/codetr_swin_transformer'

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50)
)

log_processor = dict(by_epoch=True)
