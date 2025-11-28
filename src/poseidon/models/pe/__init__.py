"""Positional encoder registry with auto-discovery."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Optional, Type

from .base import PEBase
from .registry import (
    PE_REG,
    available_pes,
    build_pe as _build_pe,
    get_pe_class,
    load_class,
    register_pe,
)

_DISCOVERED = False


def discover_pes(force: bool = False) -> None:
    """Import all PE modules so decorators can register them."""

    global _DISCOVERED
    if _DISCOVERED and not force:
        return
    package_name = __name__
    for module_info in pkgutil.iter_modules(__path__):                              
        module_name = module_info.name
        if module_name.startswith("_") or module_name in {"base", "registry"}:
            continue
        importlib.import_module(f"{package_name}.{module_name}")
    _DISCOVERED = True


def list_registered_pes() -> Dict[str, Type[PEBase]]:
    discover_pes()
    return available_pes()


def build_pe(cfg: dict, context: Optional[dict] = None) -> PEBase:
    discover_pes()
    return _build_pe(cfg, context=context)


__all__ = [
    "PEBase",
    "register_pe",
    "discover_pes",
    "list_registered_pes",
    "build_pe",
    "load_class",
    "get_pe_class",
    "PE_REG",
]
