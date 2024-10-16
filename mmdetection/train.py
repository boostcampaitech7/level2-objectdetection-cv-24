import argparse
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules 
from mmdet.registry import MODELS
from mmengine.registry import MODELS as ENGINE_MODELS
from mmdet.models import DetDataPreprocessor

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
    print(cfg.pretty_text)

    # 폴드 수와 JSON 경로 설정
    num_folds = 5
    all_fold_metrics = []

    # Runner 생성 및 학습 시작
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
