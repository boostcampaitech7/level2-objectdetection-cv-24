import numpy as np
from scipy.cluster.vq import kmeans
import json
from tqdm import tqdm
import argparse

def load_annotations(ann_file):
    with open(ann_file, 'r') as f:
        data = json.load(f)
    return data

def get_bbox_wh(annotations):
    wh = []
    for ann in tqdm(annotations['annotations'], desc="Processing annotations"):
        bbox = ann['bbox']
        w, h = bbox[2], bbox[3]
        wh.append([w, h])
    return np.array(wh)

def kmeans_anchors(wh, n_anchors, iterations=30):
    wh = wh[wh[:, 0] > 0]  # 너비가 0보다 큰 것만 선택
    wh = wh[wh[:, 1] > 0]  # 높이가 0보다 큰 것만 선택
    wh = wh[np.logical_and(wh[:, 0] < 1, wh[:, 1] < 1)]  # 1보다 작은 것만 선택 (정규화된 경우)

    s = wh.std(0)
    k, dist = kmeans(wh / s, n_anchors, iter=iterations)
    anchors = k * s
    return anchors

def anchor_fitness(anchors, wh, thr=4):
    r = wh[:, None] / anchors[None]
    x = np.minimum(r, 1 / r).min(2)
    fit = (x * (x > 1 / thr)).mean(1)
    return fit

def print_anchors(anchors):
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_idx = areas.argsort()
    anchors = anchors[sorted_idx]
    
    print("\nOptimized Anchors:")
    for i, anchor in enumerate(anchors):
        print(f"Anchor {i+1}: ({anchor[0]:.2f}, {anchor[1]:.2f})")
    
    print("\nMMYOLO Config Format:")
    print("prior_generator=dict(")
    print("    type='mmdet.YOLOAnchorGenerator',")
    print("    base_sizes=[")
    for i in range(0, len(anchors), 3):
        print("        [", end="")
        for j in range(3):
            if i+j < len(anchors):
                print(f"({anchors[i+j][0]:.2f}, {anchors[i+j][1]:.2f})", end="")
                if j < 2 and i+j+1 < len(anchors):
                    print(", ", end="")
        print("],")
    print("    ],")
    print("    strides=[8, 16, 32]")
    print(")")

def main(ann_file, n_anchors, img_size):
    annotations = load_annotations(ann_file)
    wh = get_bbox_wh(annotations)
    
    # 이미지 크기로 정규화
    wh = wh / img_size
    
    anchors = kmeans_anchors(wh, n_anchors)
    fitness = anchor_fitness(anchors, wh)
    
    print(f"Average Fitness: {fitness.mean():.4f}")
    
    # 앵커를 원래 이미지 크기로 변환
    anchors = anchors * img_size
    
    print_anchors(anchors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate optimized anchors for YOLO")
    parser.add_argument("ann_file", type=str, help="Path to the annotation file")
    parser.add_argument("--n_anchors", type=int, default=9, help="Number of anchors to generate")
    parser.add_argument("--img_size", type=int, default=1024, help="Image size")
    args = parser.parse_args()

    main(args.ann_file, args.n_anchors, args.img_size)