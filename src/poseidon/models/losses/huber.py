from __future__ import annotations

import torch
from typing import Tuple, Union

from .base import register_loss


PredType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


@register_loss(name="huber")
def huber(pred: PredType, y: torch.Tensor, delta: float = 0.05):
    if isinstance(pred, tuple):
        yhat, s = pred
        err = yhat - y
        abs_err = torch.abs(err)
        base = torch.where(abs_err < delta, 0.5 * err * err, delta * (abs_err - 0.5 * delta))
        return torch.exp(-s) * base + 0.5 * s
    err = pred - y
    abs_err = torch.abs(err)
    return torch.where(abs_err < delta, 0.5 * err * err, delta * (abs_err - 0.5 * delta))
