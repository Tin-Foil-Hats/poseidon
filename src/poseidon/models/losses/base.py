from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Union

import torch

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_REGISTRY: Dict[str, LossFn] = {}


def _register_key(key: str, fn: LossFn) -> None:
    norm = key.lower()
    existing = _REGISTRY.get(norm)
    if existing and existing is not fn:
        raise ValueError(f"loss '{norm}' already registered")
    _REGISTRY[norm] = fn


def register_loss(
    fn: Optional[LossFn] = None,
    *,
    name: Optional[str] = None,
    aliases: Optional[Iterable[str]] = None,
):
    """Decorator to register a loss function under one or more names."""

    def wrap(target: LossFn) -> LossFn:
        primary = name or getattr(target, "name", target.__name__)
        _register_key(primary, target)
        extra_aliases = aliases if aliases is not None else getattr(target, "aliases", [])
        for alias in extra_aliases:
            _register_key(alias, target)
        return target

    if fn is None:
        return wrap
    return wrap(fn)


def get_loss(name: str) -> LossFn:
    norm = name.lower()
    if norm not in _REGISTRY:
        raise KeyError(f"unknown loss '{name}'")
    return _REGISTRY[norm]


def available_losses() -> Dict[str, LossFn]:
    return dict(_REGISTRY)


def build_loss(cfg: Union[Optional[dict], str]) -> LossFn:
    if isinstance(cfg, str):
        return get_loss(cfg)
    cfg = cfg or {}
    loss_type = cfg.get("type", "mse")
    fn = get_loss(loss_type)
    if loss_type == "huber" and "delta" in cfg:
        delta = float(cfg["delta"])

        def huber_with_delta(pred, y):
            return fn(pred, y, delta=delta)                      

        return huber_with_delta
    return fn
