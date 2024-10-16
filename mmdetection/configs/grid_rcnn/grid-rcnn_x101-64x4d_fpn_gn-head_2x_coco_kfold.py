_base_ = './grid-rcnn_x101-32x4d_fpn_gn-head_2x_coco.py'

model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d'))
)

# 데이터 섹션 추가
data_root = '/data/ephemeral/home/level2-objectdetection-cv-24/kfold'  # 절대 경로 설정
classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
           'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')

# k-fold 학습을 위한 데이터 설정
data = dict(
    train=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='/fold_0_train.json',  # 첫 번째 fold의 train JSON
        img_prefix='/fold_0/train/',  # 첫 번째 fold의 train 이미지 경로
        classes=classes
    ),
    val=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='/fold_0_val.json',  # 첫 번째 fold의 val JSON
        img_prefix='/fold_0/val/',  # 첫 번째 fold의 val 이미지 경로
        classes=classes
    ),
    test=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='test.json',  # 테스트 데이터 JSON 경로 (필요시)
        img_prefix='test/',  # 테스트 이미지 경로 (필요시)
        classes=classes
    )
)
