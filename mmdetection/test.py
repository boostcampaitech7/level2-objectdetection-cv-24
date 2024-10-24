import os
import torch
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from mmdet.apis import init_detector, inference_detector
from mmengine.config import Config

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
cfg = Config.fromfile('./mmdetection/configs/custom_configs/haegang_test.py')

cfg.work_dir = './mmdetection/work_dirs/cascade_rcnn_swinL_aug2'
checkpoint_path = os.path.join(cfg.work_dir, 'best_coco_General trash_precision_epoch_15.pth')

model = init_detector(cfg, checkpoint_path, device='cuda:0')

def process_annotations(annotation_file, output_file):
    # COCO 객체 생성 및 이미지 ID 불러오기
    coco = COCO(annotation_file)
    img_ids = coco.getImgIds()

    # 결과 저장을 위한 리스트 초기화
    prediction_strings = []
    file_names = []

    # 추론 수행
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        file_names.append(img_info['file_name'])
        
        # 이미지 경로 설정
        img_path = os.path.join('/data/ephemeral/home/dataset', img_info['file_name'])
        
        # 인퍼런스 실행
        result = inference_detector(model, img_path)
        
        # 결과 포맷 설정
        prediction_string = ''
        if hasattr(result, 'pred_instances'):
            pred_instances = result.pred_instances.cpu().numpy()
            for i in range(len(pred_instances['bboxes'])):
                label = pred_instances['labels'][i]
                score = pred_instances['scores'][i]
                bbox = pred_instances['bboxes'][i]
                
                # 임계값 이상의 결과만 포함
                if score < 0.3:
                    continue
                
                # 예측 결과 포맷: "class score xmin ymin xmax ymax"
                prediction_string += f"{label} {score:.4f} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} "
        
        prediction_strings.append(prediction_string.strip())

    # 제출 파일 생성
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(output_file, index=False)

    print(f"Submission file saved at: {output_file}")
    print(submission.head())

# test.json 처리, test.json파일 경로
test_annotation_file = '/data/ephemeral/home/dataset/test.json'
test_output_file = 'thresholdsubmission_test.csv'
process_annotations(test_annotation_file, test_output_file)

# correlation.json 처리, correlation.json파일 경로
#correlation_annotation_file = '/data/ephemeral/home/dataset/correlation.json'
#correlation_output_file = 'submission_correlation.csv'
#process_annotations(correlation_annotation_file, correlation_output_file)