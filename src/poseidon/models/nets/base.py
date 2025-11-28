from __future__ import annotations

from typing import Dict, Iterable, Optional, Type

import torch.nn as nn


class NetBase(nn.Module):
    """Common base class for Poseidon nets."""

    def out_dim(self) -> int:
        return 1


_REGISTRY: Dict[str, Type[NetBase]] = {}


def _register_key(key: str, cls: Type[NetBase]) -> None:
    norm = key.lower()
    existing = _REGISTRY.get(norm)
    if existing and existing is not cls:
        raise ValueError(f"net '{norm}' already registered by {existing.__name__}")
    _REGISTRY[norm] = cls


def register_net(
    cls: Optional[Type[NetBase]] = None,
    *,
    name: Optional[str] = None,
    aliases: Optional[Iterable[str]] = None,
):
    """Decorator to register a net for lookup by name."""

    def wrap(target: Type[NetBase]) -> Type[NetBase]:
        if not issubclass(target, NetBase):
            raise TypeError("register_net requires NetBase subclasses")
        primary = name or getattr(target, "name", target.__name__)
        _register_key(primary, target)
        extra_aliases = aliases if aliases is not None else getattr(target, "aliases", [])
        for alias in extra_aliases:
            _register_key(alias, target)
        return target

    if cls is None:
        return wrap
    return wrap(cls)


def get_net_class(name: str) -> Type[NetBase]:
    norm = name.lower()
    if norm not in _REGISTRY:
        raise KeyError(f"unknown net '{name}'")
    return _REGISTRY[norm]


def available_nets() -> Dict[str, Type[NetBase]]:
    return dict(_REGISTRY)


def build_net(cfg: Optional[dict], *, in_dim: int) -> NetBase:
    cfg = cfg or {}
    net_type = cfg.get("type", "mlp")
    cls = get_net_class(net_type)
    params = {k: v for k, v in cfg.items() if k != "type"}
    return cls(in_dim=in_dim, **params)
