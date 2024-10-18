import argparse
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules 
from mmdet.registry import MODELS
from mmengine.registry import MODELS as ENGINE_MODELS
from mmdet.models import DetDataPreprocessor
import wandb
import numpy as np
import random

# sweep config 설정
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'coco/bbox_mAP'},
    'parameters': 
    {
        'batch_size': {'values': [4, 8, 16]},
        'epochs': {'values': [50, 70, 100]},
        'lr': {'max': 0.01, 'min': 0.0005}
     }
}

# sweep id 자동으로 받아오기
sweep_id = wandb.sweep(sweep=sweep_configuration, project='fixed_label_test')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detection model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='directory to save logs and models')
    parser.add_argument('--resume-from', help='checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', default='cuda', help='device used for training')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    wandb.init()
    
    lr  =  wandb.config.lr
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs
    
    # 모든 mmdetection 모듈을 등록
    register_all_modules()


    # config 파일 로드
    cfg = Config.fromfile(args.config)
    
    

    # 작업 디렉토리 설정
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # 장치 설정
    cfg.device = args.device

    # 학습 재개 설정
    if args.resume_from:
        cfg.resume = True
        cfg.load_from = args.resume_from
    else:
        cfg.resume = False
        cfg.load_from = None

    # 랜덤 시드 설정
    cfg.seed = args.seed

    # 설정 출력 (디버깅용)
    #print(cfg.pretty_text)

    # 폴드 수와 JSON 경로 설정
    num_folds = 5
    all_fold_metrics = []

    # Runner 생성 및 학습 시작
    runner = Runner.from_cfg(cfg)
    runner.log_wandb = True 
    
    runner.train()
    # Config 값 기록
    wandb.config.update({
        'learning_rate': wandb.config.get(lr),
        'batch_size': wandb.config.get(bs),
        'max_epochs': wandb.config.get(epochs)
    })    
    
if __name__ == '__main__':
    # sweep 실행
    wandb.agent(sweep_id, function=main, count=2)
