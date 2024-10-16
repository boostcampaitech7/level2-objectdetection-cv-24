import os
import shutil
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
from collections import Counter

# Load JSON
annotation = './dataset/train.json'

with open(annotation) as f:
    data = json.load(f)
    
def main():
    var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]

    X = np.ones((len(data['annotations']), 1))
    y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var])

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Check distribution
    def get_distribution(y):
        y_distr = Counter(y)
        y_vals_sum = sum(y_distr.values())
        
        return [f'{y_distr[i] / y_vals_sum:.2%}' for i in range(np.max(y) + 1)]

    distrs = [get_distribution(y)]
    index = ['training set']

    # Save k-fold data function
    def save_kfold_data(train_indices, val_indices, fold_num, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        # Image directory
        image_dir = './dataset/train'
        train_image_dir = os.path.join(output_dir, f'fold_{fold_num}/train/')
        val_image_dir = os.path.join(output_dir, f'fold_{fold_num}/val/')
        
        os.makedirs(train_image_dir, exist_ok=True)
        os.makedirs(val_image_dir, exist_ok=True)

        # Train, val data 저장소
        train_data = {'annotations': [], 'images': []}
        val_data = {'annotations': [], 'images': []}

        for idx in train_indices:
            ann = data['annotations'][idx]
            train_data['annotations'].append(ann)
            # 4자리 문자열로 변환
            image_id = f"{int(ann['image_id']):04d}"  # 앞에 0을 붙여서 4자리로 만듦
            # Copy images
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            shutil.copy(image_path, train_image_dir)

        for idx in val_indices:
            ann = data['annotations'][idx]
            val_data['annotations'].append(ann)
            # 4자리 문자열로 변환
            image_id = f"{int(ann['image_id']):04d}"
            # Copy images
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            shutil.copy(image_path, val_image_dir)

        # Save JSON files
        with open(os.path.join(output_dir, f'fold_{fold_num}_train.json'), 'w') as f:
            json.dump(train_data, f)

        with open(os.path.join(output_dir, f'fold_{fold_num}_val.json'), 'w') as f:
            json.dump(val_data, f)

    # Loop through folds
    for fold_ind, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        train_y, val_y = y[train_idx], y[val_idx]
        train_gr, val_gr = groups[train_idx], groups[val_idx]
        
        assert len(set(train_gr) & set(val_gr)) == 0
        
        distrs.append(get_distribution(train_y))
        distrs.append(get_distribution(val_y))
        
        index.append(f'train - fold{fold_ind}')
        index.append(f'val - fold{fold_ind}')

        # Save data for this fold
        save_kfold_data(train_idx, val_idx, fold_ind, output_dir)

    categories = [d['name'] for d in data['categories']]
    print(pd.DataFrame(distrs, index=index, columns=[categories[i] for i in range(np.max(y) + 1)]))

if __name__ == '__main__':
    output_dir = './kfold/'
    main()