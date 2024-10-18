import json
import random

def split_coco_annotation(input_json, train_json, val_json, split_ratio=0.8):
    with open(input_json, 'r') as f:
        coco = json.load(f)
    
    # 이미지 ID를 기준으로 섞기
    image_ids = [img['id'] for img in coco['images']]
    random.shuffle(image_ids)
    
    # 분할
    split_index = int(len(image_ids) * split_ratio)
    train_ids = set(image_ids[:split_index])
    val_ids = set(image_ids[split_index:])

    # 학습과 검증 데이터셋 생성
    train_images = [img for img in coco['images'] if img['id'] in train_ids]
    val_images = [img for img in coco['images'] if img['id'] in val_ids]
    
    train_annotations = [ann for ann in coco['annotations'] if ann['image_id'] in train_ids]
    val_annotations = [ann for ann in coco['annotations'] if ann['image_id'] in val_ids]

    # 새로운 COCO 형식으로 JSON 파일 생성
    train_coco = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': coco['categories']
    }
    val_coco = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': coco['categories']
    }

    # 파일 저장
    with open(train_json, 'w') as f:
        json.dump(train_coco, f)
    with open(val_json, 'w') as f:
        json.dump(val_coco, f)

# 사용 예시
split_coco_annotation('./dataset/train_copy.json', './dataset/label_train.json', './dataset/label_val.json', split_ratio=0.8)
