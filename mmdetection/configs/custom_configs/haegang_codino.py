_base_ = [
    '../../projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_lsj_16xb1_3x_coco.py'
]



dataset_type = 'CocoDataset'
data_root = '../dataset/'
classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
           'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')
train_dataloader = dict(
    batch_size=1,  # 기존 설정 유지
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='temp_train.json',
        metainfo=dict(classes=classes),  # 클래스 정보 추가
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='RandomResize',
                scale=(1280, 1280),
                ratio_range=(0.1, 2.0),
                keep_ratio=True
            ),
            dict(
                type='RandomCrop',
                crop_type='absolute_range',
                crop_size=(1280, 1280),
                recompute_bbox=True,
                allow_negative_crop=True
            ),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
            dict(type='RandomFlip', prob=0.5),
            dict(type='Pad', size=(1280, 1280), pad_val=dict(img=(114, 114, 114))),
        ]
    )
)

# 검증 데이터 로더 설정
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='temp_val.json',
        metainfo=dict(classes=classes),  # 클래스 정보 추가
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1280, 1280), keep_ratio=True),
            dict(type='Pad', size=(1280, 1280), pad_val=dict(img=(114, 114, 114))),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
            )
        ]
    )
)

test_dataloader = val_dataloader  # 검증 파이프라인과 동일하게 사용

# ReduceOnPlateau 스케줄러 설정
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=50,  
        eta_min=1e-6,  
        by_epoch=True  
    )
]
