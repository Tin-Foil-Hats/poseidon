from __future__ import annotations

import torch
from typing import Tuple, Union

from .base import register_loss


PredType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


@register_loss(name="mae")
def mae(pred: PredType, y: torch.Tensor):
    if isinstance(pred, tuple):
        yhat, s = pred
        return torch.exp(-s) * torch.abs(yhat - y) + 0.5 * s
    return torch.abs(pred - y)
