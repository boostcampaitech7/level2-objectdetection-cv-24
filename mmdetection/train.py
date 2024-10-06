# 모듈 import
import wandb
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
cfg = Config.fromfile('./mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py')

root='./dataset/'

# dataset config 수정
cfg.data.train.classes = classes
cfg.data.train.img_prefix = root
cfg.data.train.ann_file = root + 'train.json' # train json 정보
cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = root + 'test.json' # test json 정보
cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

cfg.data.samples_per_gpu = 4

cfg.seed = 2022
cfg.gpu_ids = [0]
cfg.work_dir = './mmdetection/work_dirs/cascade_rcnn_r50_fpn_1x_coco_trash'

if isinstance(cfg.model.roi_head.bbox_head, list):
    for head in cfg.model.roi_head.bbox_head:
        head.num_classes = 10
else:
    cfg.model.roi_head.bbox_head.num_classes = 10

# cfg.model.roi_head.bbox_head.num_classes = 10
# cfg.model.bbox_head.num_classes = 10 # Retinanet

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.device = get_device()

wandb.init(project='mmdetection-trash-detection', entity='gaemanssi2-naver-ai-boostcamp', name=cfg.model.type)

wandb.config.update({
    "model": cfg.model.type,
    "backbone": cfg.model.backbone.type,
    "learning_rate": cfg.optimizer.lr,
    "batch_size": cfg.data.samples_per_gpu,
    "num_epochs": cfg.runner.max_epochs,
    "img_scale": cfg.data.train.pipeline[2]['img_scale'],
    "classes": classes
})

if 'log_config' not in cfg:
    cfg.log_config = dict(interval=50)
if 'hooks' not in cfg.log_config:
    cfg.log_config.hooks = []

cfg.evaluation = dict(interval=1, metric='bbox')

# build_dataset
datasets = [build_dataset(cfg.data.train)]

# dataset 확인
datasets[0]

# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()

cfg.log_config.hooks.append(
    dict(type='MMDetWandbHook',
         init_kwargs={'project': 'mmdetection-trash-detection'},
         interval=len(datasets[0]),
         log_checkpoint=True,
         log_checkpoint_metadata=True,
         num_eval_images=0) # val dataset을 안쓰는 경우 0, 사용 시 100 정도로 숫자 조정
)

# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=False)

wandb.finish()