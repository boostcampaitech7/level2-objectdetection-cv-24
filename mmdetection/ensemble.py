import pandas as pd
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
import csv

def read_predictions(csv_file):
    df = pd.read_csv(csv_file) 
    # 'PredictionString' 열의 NaN 값을 빈 문자열로 채우고 문자열 타입으로 변환
    df['PredictionString'] = df['PredictionString'].fillna('').astype(str)
    return df

def parse_prediction_string(pred_string):
    if pd.isna(pred_string) or pred_string == 'nan':
        return np.array([]), np.array([]), np.array([])
    
    pred_list = pred_string.split()
    boxes = []
    scores = []
    labels = []
    for i in range(0, len(pred_list), 6):
        try:
            label = int(float(pred_list[i]))
            score = float(pred_list[i+1])
            x1, y1, x2, y2 = map(float, pred_list[i+2:i+6])
            
            # 좌표 검증 및 수정
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # 최소 크기 설정
            if x2 - x1 < 1:
                x2 = x1 + 1
            if y2 - y1 < 1:
                y2 = y1 + 1
            
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            labels.append(label)
        except ValueError:
            print(f"Warning: Could not parse prediction: {pred_list[i:i+6]}")
            continue
    return np.array(boxes), np.array(scores), np.array(labels)

def perform_wbf(predictions, image_size=(1024, 1024), iou_thr=0.5, skip_box_thr=0.0001, weights=None):
    boxes_list = []
    scores_list = []
    labels_list = []
    
    for pred in predictions:
        boxes, scores, labels = parse_prediction_string(pred)
        
        if len(boxes) > 0:
            # 박스 정규화
            boxes[:, [0, 2]] /= image_size[0]
            boxes[:, [1, 3]] /= image_size[1]
            
            # 0과 1 사이로 클리핑
            boxes = np.clip(boxes, 0, 1)
            
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
    
    if len(boxes_list) > 0:
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, weights=weights,
            iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )
        
        # 박스 역정규화
        boxes[:, [0, 2]] *= image_size[0]
        boxes[:, [1, 3]] *= image_size[1]
    else:
        # 예측 결과가 없는 경우 빈 배열 반환
        boxes = np.array([])
        scores = np.array([])
        labels = np.array([])

    return boxes, scores, labels

def format_predictions(boxes, scores, labels):
    pred_strings = []
    for box, score, label in zip(boxes, scores, labels):
        pred_strings.append(f"{int(label)} {score:.6f} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
    return " ".join(pred_strings)

def ensemble_predictions(csv_files, output_file, weights=None, iou_thr=0.5, skip_box_thr=0.00001):
    all_predictions = {}
    
    for csv_file in csv_files:
        df = read_predictions(csv_file)
        for _, row in df.iterrows():
            image_id = row['image_id']
            if image_id not in all_predictions:
                all_predictions[image_id] = []
            all_predictions[image_id].append(row['PredictionString'])
    
    results = []
    for image_id, predictions in all_predictions.items():
        # weights를 예측 수에 맞게 조정
        adjusted_weights = weights[:len(predictions)] if weights else None
        boxes, scores, labels = perform_wbf(predictions, weights=adjusted_weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        prediction_string = format_predictions(boxes, scores, labels)
        results.append({'PredictionString': prediction_string, 'image_id': image_id})
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"Ensemble results saved to {output_file}")

def parse_prediction_string(pred_string):
    if pd.isna(pred_string) or pred_string == '' or pred_string == 'nan':
        return np.array([]), np.array([]), np.array([])
    
    pred_list = pred_string.split()
    boxes = []
    scores = []
    labels = []
    for i in range(0, len(pred_list), 6):
        try:
            label = int(float(pred_list[i]))
            score = float(pred_list[i+1])
            x1, y1, x2, y2 = map(float, pred_list[i+2:i+6])
            
            # 좌표 검증 및 수정
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # 최소 크기 설정
            if x2 - x1 < 1:
                x2 = x1 + 1
            if y2 - y1 < 1:
                y2 = y1 + 1
            
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            labels.append(label)
        except (ValueError, IndexError):
            print(f"Warning: Could not parse prediction: {pred_list[i:i+6]}")
            continue
    return np.array(boxes), np.array(scores), np.array(labels)

# 사용 예시
csv_files = ['/data/ephemeral/home/eva_test/mmdetection/checkpoints/grid_rcnn.csv',
             '/data/ephemeral/home/eva_test/mmdetection/checkpoints/retianet_r50.csv',
             '/data/ephemeral/home/eva_test/mmdetection/checkpoints/align_detr.csv',
             '/data/ephemeral/home/eva_test/mmdetection/checkpoints/cascade_rcnn.csv',
             '/data/ephemeral/home/eva_test/mmdetection/checkpoints/fcos_x101.csv']
output_file = 'ensemble_predictions.csv'
weights = [1, 1, 1, 1, 1]  # 각 모델에 동일한 가중치 적용

ensemble_predictions(csv_files, output_file, weights=weights)