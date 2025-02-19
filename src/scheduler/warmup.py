from typing import List

import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):
    """
    The WarmupLR scheduler
    from https://arxiv.org/pdf/1706.03762
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 3000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            (
                lr
                * self.warmup_steps**0.5
                * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
            )
            for lr in self.base_lrs
        ]
