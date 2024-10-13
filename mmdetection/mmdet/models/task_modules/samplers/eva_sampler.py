import torch
from mmdet.models.task_modules.samplers import BaseSampler
from mmdet.structures.bbox import BaseBoxes
from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.samplers import SamplingResult

@TASK_UTILS.register_module()
class EVASampler(BaseSampler):
    def __init__(self, num, pos_fraction, neg_pos_ub=-1, add_gt_as_proposals=True, **kwargs):
        super().__init__(num, pos_fraction, neg_pos_ub, add_gt_as_proposals)

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if neg_inds.numel() <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)
        
    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.

        Args:
            gallery (Tensor): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor: sampled indices.
        """
        assert len(gallery) >= num
        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            gallery = torch.tensor(
                gallery, dtype=torch.long, device=torch.cuda.current_device())
        perm = torch.randperm(gallery.numel())[:num].cuda()
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def sample(self, assign_result, pred_instances, gt_instances, **kwargs):
        priors = pred_instances.priors
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        if len(priors.shape) < 2:
            priors = priors[None, :]

        gt_flags = priors.new_zeros((priors.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if isinstance(gt_bboxes, BaseBoxes):
                gt_bboxes_ = gt_bboxes.tensor
            else:
                gt_bboxes_ = gt_bboxes
            
            if isinstance(priors, BaseBoxes):
                priors_ = priors.tensor
            else:
                priors_ = priors
            
            priors = torch.cat([gt_bboxes_, priors_], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = priors.new_ones(gt_bboxes_.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self._sample_pos(assign_result, num_expected_pos, bboxes=priors, **kwargs)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self._sample_neg(assign_result, num_expected_neg, bboxes=priors, **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(pos_inds, neg_inds, priors, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result