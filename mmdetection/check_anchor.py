import json
import numpy as np
from sklearn.cluster import KMeans

# COCO 형식의 annotation 파일 경로
ann_file = './dataset/train.json'

# JSON 파일 로드
with open(ann_file) as f:
    data = json.load(f)

# 각 객체의 너비와 높이 계산
widths = []
heights = []
for annotation in data['annotations']:
    bbox = annotation['bbox']
    width = bbox[2]  # bbox format [x, y, width, height]
    height = bbox[3]
    widths.append(width)
    heights.append(height)

# 너비와 높이 평균 및 중앙값
widths = np.array(widths)
heights = np.array(heights)
print(f'Width mean: {np.mean(widths)}, median: {np.median(widths)}')
print(f'Height mean: {np.mean(heights)}, median: {np.median(heights)}')

# 객체 크기 배열
boxes = np.array(list(zip(widths, heights)))

# 9개의 클러스터(앵커) 생성
kmeans = KMeans(n_clusters=9, random_state=0).fit(boxes)

# 클러스터의 중심이 새로운 앵커 크기
anchors = kmeans.cluster_centers_
print('Anchors:', anchors)

# 각 스케일(작은, 중간, 큰)에 맞게 앵커 크기 분배
anchors = sorted(anchors, key=lambda x: x[0] * x[1])  # 면적 기준으로 정렬
small_anchors = anchors[:3]   # 작은 스케일
medium_anchors = anchors[3:6] # 중간 스케일
large_anchors = anchors[6:]   # 큰 스케일

# YOLO 앵커 형식에 맞게 출력
print(f'Small anchors: {small_anchors}')
print(f'Medium anchors: {medium_anchors}')
print(f'Large anchors: {large_anchors}')