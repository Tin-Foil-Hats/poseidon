"""Net registry and auto-discovery helpers."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Optional, Type

from .base import NetBase, available_nets, build_net as _build_net, get_net_class, register_net

_DISCOVERED = False


def discover_nets(force: bool = False) -> None:
    """Import all net modules so decorators can register them."""

    global _DISCOVERED
    if _DISCOVERED and not force:
        return
    package_name = __name__
    for module_info in pkgutil.iter_modules(__path__):                              
        module_name = module_info.name
        if module_name.startswith("_") or module_name in {"base", "nets"}:
            continue
        importlib.import_module(f"{package_name}.{module_name}")
    _DISCOVERED = True


def list_registered_nets() -> Dict[str, Type[NetBase]]:
    discover_nets()
    return available_nets()


def build_net(cfg: Optional[dict], *, in_dim: int) -> NetBase:
    discover_nets()
    return _build_net(cfg, in_dim=in_dim)


__all__ = [
    "NetBase",
    "register_net",
    "discover_nets",
    "list_registered_nets",
    "build_net",
    "get_net_class",
]
