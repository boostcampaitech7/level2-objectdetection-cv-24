# custom_hooks.py
from typing import Optional
from mmengine.hooks import Hook
import wandb
from mmdet.registry import HOOKS

from mmengine.runner import load_checkpoint
from mmengine.logging import print_log

@HOOKS.register_module()
class WandbLoggerHook(Hook):
    def __init__(self, init_kwargs=None):
        super().__init__()
        self.init_kwargs = init_kwargs or {}

    def before_run(self, runner):
        wandb.init(**self.init_kwargs)
        runner.logger.info("Wandb initialized")
        
    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        
        # outputs에서 loss 값을 추출하여 wandb에 기록
        if outputs is not None and 'loss' in outputs:
            loss_value = outputs['loss'].item()
            wandb.log({'loss': loss_value, 'batch_idx': batch_idx})
                  
    def after_run(self, runner):
        wandb.finish()
        runner.logger.info("Wandb run finished")


@HOOKS.register_module()
class CustomLoadCheckpointHook(Hook):
    def before_run(self, runner):
        """Checkpoint를 로드한 후 불필요한 레이어를 초기화합니다."""
        if runner.cfg.get('load_from'):
            print_log(
                f"Loading checkpoint from {runner.cfg.load_from} with strict=False", 
                runner.logger
            )

            # strict=False로 checkpoint 로드
            load_checkpoint(runner.model, runner.cfg.load_from, strict=False)

            print_log("Checkpoint loaded. Initializing specific layers...", runner.logger)

            # 불일치하는 레이어들을 수동으로 초기화
            self.initialize_layers(runner.model)

    def initialize_layers(self, model):
        """모델의 모든 헤드와 관련된 레이어 초기화."""
        # Query head 초기화
        if hasattr(model, 'query_head'):
            for i, branch in enumerate(model.query_head.cls_branches):
                try:
                    branch.reset_parameters()
                    print_log(f"Initialized query_head.cls_branches[{i}]", logger=None)
                except AttributeError:
                    print_log(f"query_head.cls_branches[{i}] does not support reset_parameters.", logger=None)

        # ROI head 초기화
        if hasattr(model, 'roi_head'):
            try:
                model.roi_head[0].bbox_head.fc_cls.reset_parameters()
                model.roi_head[0].bbox_head.fc_reg.reset_parameters()
                print_log("Initialized ROI head's layers", logger=None)
            except AttributeError:
                print_log("ROI head does not support reset_parameters.", logger=None)

        # BBox head와 ATSS 관련 레이어 초기화
        if hasattr(model, 'bbox_head'):
            try:
                model.bbox_head[0].atss_cls.reset_parameters()
                print_log("Initialized bbox_head.atss_cls", logger=None)
            except AttributeError:
                print_log("bbox_head.atss_cls does not support reset_parameters.", logger=None)

        # 기타 모든 관련 모듈 초기화
        for name, module in model.named_modules():
            if any(keyword in name for keyword in ['cls', 'reg', 'atss']):
                try:
                    module.reset_parameters()
                    print_log(f"Initialized {name}", logger=None)
                except AttributeError:
                    print_log(f"{name} does not support reset_parameters.", logger=None)
