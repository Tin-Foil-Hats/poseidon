from __future__ import annotations

import torch
from typing import Tuple, Union

from .base import register_loss


PredType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


@register_loss(name="mse")
def mse(pred: PredType, y: torch.Tensor):
    if isinstance(pred, tuple):
        yhat, s = pred
        return 0.5 * torch.exp(-s) * (yhat - y) ** 2 + 0.5 * s
    return (pred - y) ** 2
