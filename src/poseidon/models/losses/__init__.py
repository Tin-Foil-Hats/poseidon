"""Loss registry with auto-discovery."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Optional, Union

from .base import available_losses, build_loss as _build_loss, get_loss, register_loss

_DISCOVERED = False


def discover_losses(force: bool = False) -> None:
    global _DISCOVERED
    if _DISCOVERED and not force:
        return
    package_name = __name__
    for module_info in pkgutil.iter_modules(__path__):                              
        module_name = module_info.name
        if module_name.startswith("_") or module_name == "base":
            continue
        importlib.import_module(f"{package_name}.{module_name}")
    _DISCOVERED = True


def list_registered_losses() -> Dict[str, callable]:
    discover_losses()
    return available_losses()


def build_loss(cfg: Union[Optional[dict], str]):
    discover_losses()
    return _build_loss(cfg)


__all__ = [
    "register_loss",
    "discover_losses",
    "list_registered_losses",
    "build_loss",
    "get_loss",
]
