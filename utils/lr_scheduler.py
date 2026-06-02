import math
import torch
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWithWarmup(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        warmup_ratio: float = 0.1,
        eta_min=0.0,
        last_epoch: int = -1,
    ):
        assert warmup_ratio <= 1, "warmup_ratio must be less or equal than 1."

        self.T_max = T_max
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = int(warmup_ratio * T_max)
        self.cosine_steps = T_max - self.warmup_steps
        self.eta_min = eta_min  # min relative learning rate

        self.max_lrs = [group["lr"] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.get_closed_form_lr(self.last_epoch)

    def get_closed_form_lr(self, epoch):
        calculated_lrs = []
        if self.warmup_steps > 0 and epoch < self.warmup_steps:
            # --- Warmup Phase ---
            for max_lr in self.max_lrs:
                calculated_lrs.append(max_lr * (epoch + 1) / self.warmup_steps)
        else:
            # --- Cosine Annealing Phase ---
            progress = max(1, min(epoch - self.warmup_steps + 1, self.cosine_steps))
            progress = progress / (self.cosine_steps + 1)

            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
            lr_scale = self.eta_min + (1 - self.eta_min) * lr_scale

            for max_lr in self.max_lrs:
                calculated_lrs.append(max_lr * lr_scale)

        return calculated_lrs
