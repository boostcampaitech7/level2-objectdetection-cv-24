from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.single_stage import SingleStageDetector

@MODELS.register_module()
class YOLOv5(SingleStageDetector):
    """Implementation of YOLOv5."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def loss(self, batch_inputs, batch_data_samples):
        """Implement forward function of YOLOv5."""
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses
    
    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        """Implement forward function of YOLOv5."""
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        return results_list

    def _forward(self, batch_inputs, batch_data_samples=None, mode='tensor'):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        if mode == 'tensor':
            img = batch_inputs
            feats = self.extract_feat(img)
            return self.bbox_head.forward(feats)
        elif mode == 'predict':
            return self.predict(batch_inputs, batch_data_samples)
        elif mode == 'loss':
            return self.loss(batch_inputs, batch_data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')