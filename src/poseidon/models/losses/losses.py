"""Backward-compatible shim pointing to the loss registry package."""

from __future__ import annotations

from . import build_loss, discover_losses, get_loss, list_registered_losses, register_loss

__all__ = [
    "register_loss",
    "discover_losses",
    "list_registered_losses",
    "build_loss",
    "get_loss",
]
