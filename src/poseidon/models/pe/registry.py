from __future__ import annotations

import importlib
from typing import Dict, Iterable, Optional, Type

from .base import PEBase

_REGISTRY: Dict[str, Type[PEBase]] = {}
_ALIASES: Dict[str, str] = {}


def _register_key(key: str, cls: Type[PEBase]) -> None:
    norm = key.lower()
    existing = _REGISTRY.get(norm)
    if existing and existing is not cls:
        raise ValueError(f"PE '{norm}' already registered by {existing.__name__}")
    _REGISTRY[norm] = cls
    _ALIASES.setdefault(norm, norm)


def register_pe(
    cls: Optional[Type[PEBase]] = None,
    *,
    name: Optional[str] = None,
    aliases: Optional[Iterable[str]] = None,
):
    """Decorator to register a positional encoder implementation."""

    def wrap(target: Type[PEBase]) -> Type[PEBase]:
        if not issubclass(target, PEBase):
            raise TypeError("register_pe requires PEBase subclasses")
        primary = (name or getattr(target, "name", target.__name__)).lower()
        _register_key(primary, target)
        extra_aliases = aliases if aliases is not None else getattr(target, "aliases", [])
        for alias in extra_aliases:
            alias_norm = alias.lower()
            existing = _ALIASES.get(alias_norm)
            if existing and existing != primary:
                raise ValueError(f"alias '{alias}' already mapped to '{existing}'")
            _ALIASES[alias_norm] = primary
        return target

    if cls is None:
        return wrap
    return wrap(cls)


def get_pe_class(name: str) -> Type[PEBase]:
    norm = name.lower()
    canonical = _ALIASES.get(norm, norm)
    if canonical not in _REGISTRY:
        raise KeyError(f"unknown positional encoder '{name}'")
    return _REGISTRY[canonical]


def available_pes() -> Dict[str, Type[PEBase]]:
    return dict(_REGISTRY)


def load_class(path: str) -> Type[PEBase]:
    mod, cls_name = path.split(":")
    if mod.startswith("locenc.pe"):
        mod = mod.replace("locenc.pe", "poseidon.models.pe", 1)
    module = importlib.import_module(mod)
    return getattr(module, cls_name)


def build_pe(cfg: dict, context: Optional[dict] = None) -> PEBase:
    if cfg is None:
        raise ValueError("PE cfg must be provided")

    if "target" in cfg:
        cls = load_class(cfg["target"])
        init_kwargs = cfg.get("init", {})
    else:
        if "type" not in cfg:
            raise ValueError("PE cfg must include 'type' or 'target'.")
        cls = get_pe_class(cfg["type"])
        init_kwargs = {k: v for k, v in cfg.items() if k != "type"}

    pe = cls(**init_kwargs)
    if hasattr(pe, "bind_context") and context is not None:
        pe.bind_context(context)
    return pe


PE_REG = _REGISTRY
