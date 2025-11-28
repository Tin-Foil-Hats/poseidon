"""Learning-rate schedulers used by Poseidon training loops."""

from __future__ import annotations

import math
from typing import Iterable, List

from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Linear warmup to the base LR followed by cosine annealing.

    This mirrors the scheduler used in earlier internal configurations where a
    fixed number of warmup epochs ramp the learning rate from
    ``warmup_start_lr`` to the optimizer's base learning rate before a cosine
    decay towards ``eta_min``. The scheduler expects to be stepped once per
    epoch.
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if warmup_epochs > max_epochs:
            raise ValueError("warmup_epochs cannot exceed max_epochs")

        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        self.warmup_start_lr = float(warmup_start_lr)
        self.eta_min = float(eta_min)

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            raise RuntimeError(
                "LinearWarmupCosineAnnealingLR.get_lr() should be called via step(); "
                "call scheduler.step() to update the learning rate."
            )

        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            return self._get_warmup_lrs()
        return self._get_cosine_lrs()

                                                                          
    def _get_closed_form_lr(self) -> List[float]:
        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            return self._get_warmup_lrs()
        return self._get_cosine_lrs()

    def _get_warmup_lrs(self) -> List[float]:
        if self.warmup_epochs == 0:
            return [base_lr for base_lr in self.base_lrs]

        if self.warmup_epochs == 1:
                                                                                         
            if self.last_epoch == 0:
                return [self.warmup_start_lr for _ in self.base_lrs]
            return [base_lr for base_lr in self.base_lrs]

        step = self.last_epoch

        if step <= 0:
            return [self.warmup_start_lr for _ in self.base_lrs]

        progress = min(step, self.warmup_epochs - 1) / max(1, self.warmup_epochs - 1)
        return [
            self.warmup_start_lr + progress * (base_lr - self.warmup_start_lr)
            for base_lr in self.base_lrs
        ]

    def _get_cosine_lrs(self) -> List[float]:
        if self.warmup_epochs > 0:
            cosine_length = max(1, self.max_epochs - self.warmup_epochs)
            effective = max(0, self.last_epoch - self.warmup_epochs + 1)
        else:
            cosine_length = max(1, self.max_epochs - 1)
            effective = max(0, self.last_epoch)

        progress = min(effective, cosine_length) / cosine_length
        return [
            self.eta_min
            + 0.5 * (base_lr - self.eta_min) * (1.0 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


__all__: Iterable[str] = ["LinearWarmupCosineAnnealingLR"]
