_base_ = './cascade-rcnn_r50_fpn_20e_coco.py'
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192, 
        depths=[2, 2, 18, 2], 
        num_heads=[6, 12, 24, 48],  
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth')  # Swin Large Pretrained 모델
    ),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],  
        out_channels=256,
        num_outs=5)
)