"""Backward compatible shim that re-exports the new registry API."""

from __future__ import annotations

from . import NetBase, build_net, discover_nets, get_net_class, list_registered_nets, register_net

__all__ = [
    "NetBase",
    "register_net",
    "discover_nets",
    "list_registered_nets",
    "build_net",
    "get_net_class",
]
