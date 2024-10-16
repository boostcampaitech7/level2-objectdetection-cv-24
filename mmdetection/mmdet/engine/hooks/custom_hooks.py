# custom_hooks.py
from typing import Optional
from mmengine.hooks import Hook
import wandb
from mmdet.registry import HOOKS

@HOOKS.register_module()
class WandbLoggerHook(Hook):
    def __init__(self, init_kwargs=None):
        super().__init__()
        self.init_kwargs = init_kwargs or {}

    def before_run(self, runner):
        wandb.init(**self.init_kwargs)
        runner.logger.info("Wandb initialized")
        
        # Config 값 기록
        wandb.config.update({
            'learning_rate': runner.cfg.optimizer.lr,  # optimizer에서 learning_rate 가져오기
            'batch_size': runner.cfg.data.samples_per_gpu,  # batch_size 가져오기
            'max_epochs': runner.cfg.total_epochs,  # total_epochs 가져오기
        })

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        # outputs에서 loss 값을 추출하여 wandb에 기록
        if outputs is not None and 'loss' in outputs:
            loss_value = outputs['loss'].item()  # loss 값을 추출
            wandb.log({'loss': loss_value, 'batch_idx': batch_idx})  # wandb에 batch index와 함께 기록
            # runner.logger.info(f"Logged loss: {loss_value} for batch: {batch_idx}")

    def after_run(self, runner):
        wandb.finish()
        runner.logger.info("Wandb run finished")
