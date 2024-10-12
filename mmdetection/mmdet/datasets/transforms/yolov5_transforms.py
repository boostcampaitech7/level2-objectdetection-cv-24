import random
import numpy as np
from mmdet.registry import TRANSFORMS
import copy
import mmcv

@TRANSFORMS.register_module()
class CustomMosaic:
    def __init__(self, img_scale=(640, 640), center_ratio_range=(0.5, 1.5),
                 prob=1.0, pad_val=114.0, bbox_clip_border=True):
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.prob = prob
        self.pad_val = pad_val
        self.bbox_clip_border = bbox_clip_border

    def __call__(self, results):
        """Apply mosaic augmentation."""
        if random.random() > self.prob:
            return results

        results = self._mosaic_transform(results)
        
        # Ensure mix_results is present in the results
        if 'mix_results' not in results:
            results['mix_results'] = []

        return results

    def _mosaic_transform(self, results):
        """Mosaic transform function."""
        assert 'mix_results' not in results, "'mix_results' in results"
        mosaic_labels = []
        mosaic_bboxes = []
        
        # Create mosaic image
        mosaic_img = np.full(
            (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
            self.pad_val,
            dtype=results['img'].dtype)
        
        # Center coordinates
        center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                # In this implementation, we need to ensure that we have enough images
                # You might want to modify this part based on your dataset
                results_patch = copy.deepcopy(results)  # TODO :여기서 다른 이미지를 선택하도록 수정 필요

            if 'annotations' not in results_patch:
                results_patch['annotations'] = []

            if 'gt_labels' not in results_patch:
                annotations = results_patch.get('annotations', [])
                gt_labels = [ann['category_id'] for ann in annotations]
                results_patch['gt_labels'] = np.array(gt_labels, dtype=np.int64)

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # 이미지 크기 맞추기 keep_ratio resize
            scale_ratio_i = min(self.img_scale[1] / h_i, self.img_scale[0] / w_i)
            img_i = mmcv.imresize(img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_labels_i = results_patch['gt_labels']

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)

            if self.bbox_clip_border:
                mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0, 2 * self.img_scale[0])
                mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0, 2 * self.img_scale[1])

            valid_inds = (mosaic_bboxes[:, 2] > mosaic_bboxes[:, 0]) & \
                        (mosaic_bboxes[:, 3] > mosaic_bboxes[:, 1])
            mosaic_bboxes = mosaic_bboxes[valid_inds]
            mosaic_labels = mosaic_labels[valid_inds]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_labels'] = mosaic_labels

        results['mix_results'] = [copy.deepcopy(results)]

        return results

    def _mosaic_combine(self, loc, center_position, img_shape):
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position (Sequence[float]): Center coordinates of mosaic
              image.
            img_shape (Sequence[int]): Image shape as (h, w).

        Returns:
            tuple[tuple[int]]: Corresponding coordinate of pasting and
              cropping
              - paste_coord (tuple): paste corner coordinate in mosaic image.
              - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        h, w = img_shape
        # calculate coordinate
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position[0] - w, 0), \
                             max(center_position[1] - h, 0), \
                             center_position[0], \
                             center_position[1]
            crop_coord = w - (x2 - x1), h - (y2 - y1), w, h
        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position[0], \
                             max(center_position[1] - h, 0), \
                             min(center_position[0] + w, self.img_scale[0] * 2), \
                             center_position[1]
            crop_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position[0] - w, 0), \
                             center_position[1], \
                             center_position[0], \
                             min(self.img_scale[1] * 2, center_position[1] + h)
            crop_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position[0], \
                             center_position[1], \
                             min(center_position[0] + w, self.img_scale[0] * 2), \
                             min(self.img_scale[1] * 2, center_position[1] + h)
            crop_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord