from mmengine.hooks import Hook

class EarlyStoppingHook(Hook):
    def __init__(self, monitor='bbox_mAP', min_delta=0.001, patience=5, rule='greater'):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.rule = rule
        self.wait = 0
        self.best_score = None

    def before_train_epoch(self, runner):
        if self.best_score is None:
            self.best_score = -float('inf') if self.rule == 'greater' else float('inf')

    def after_val_epoch(self, runner, metrics):
        current_score = metrics.get(self.monitor)
        if current_score is None:
            runner.logger.warning(f'Metric {self.monitor} not found.')
            return

        if self.rule == 'greater':
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta

        if improved:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            runner.logger.info(f'Early stopping at epoch {runner.epoch + 1}')
            runner.should_stop = True
