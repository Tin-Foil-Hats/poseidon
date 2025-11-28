import math
from typing import Dict, List, Tuple, Union

import torch


def deg2rad(x: torch.Tensor) -> torch.Tensor:
    return x * (math.pi / 180.0)


def wrap_lon_pi(lon_rad: torch.Tensor) -> torch.Tensor:
    return (lon_rad + math.pi) % (2 * math.pi) - math.pi


def wrap_lon_pi_float(lon_rad: float) -> float:
    """Wrap a scalar longitude in radians to the [-pi, pi) interval."""

    return ((lon_rad + math.pi) % (2 * math.pi)) - math.pi


def normalize_lon_bounds(lon_min_deg: float, lon_max_deg: float) -> Tuple[float, float, bool]:
    """Return wrapped longitude bounds and whether the interval crosses the seam.

    The inputs are expressed in degrees. The outputs are radians in the same
    [-pi, pi) frame used by :func:`wrap_lon_pi`. When the original interval
    spans the 180-degree seam (e.g., 170° to 190°), the upper bound is shifted
    by +2π so that ``lon_max`` remains numerically greater than ``lon_min``.
    The boolean flag indicates that this shift occurred, and downstream code
    should mirror the adjustment for samples (by adding 2π to wrapped values
    below ``lon_min``).
    """

    lon_min = wrap_lon_pi_float(math.radians(lon_min_deg))
    lon_max = wrap_lon_pi_float(math.radians(lon_max_deg))
    crosses_seam = False
    if lon_max < lon_min:
        lon_max += 2 * math.pi
        crosses_seam = True
    return lon_min, lon_max, crosses_seam


def _coerce_stats(stats: dict | None) -> dict | None:
    if stats is None:
        return None
    if isinstance(stats, dict):
        out = {k: stats.get(k, stats.get(k.upper())) for k in ("mean", "std", "min", "max")}
        if out["min"] is None or out["max"] is None:
            return {k: v for k, v in stats.items() if v is not None}
        return out
    if isinstance(stats, (tuple, list)) and len(stats) >= 4:
        keys = ("mean", "std", "min", "max")
        return {k: stats[i] for i, k in enumerate(keys) if i < len(stats)}
    if isinstance(stats, (tuple, list)) and len(stats) == 2:
        return {"mean": stats[0], "std": stats[1]}
    return None


def time_normalize(t_sec: torch.Tensor | None, mode: str, stats: dict | None) -> torch.Tensor | None:
    stats = _coerce_stats(stats)
    if t_sec is None or stats is None or mode == "none":
        return t_sec
    mode = mode.lower()
    if mode == "center":
        mu = float(stats.get("mean", 0.0))
        return t_sec - mu
    if mode == "zscore":
        mu = float(stats.get("mean", 0.0))
        sd = float(stats.get("std", 1.0))
        return (t_sec - mu) / (sd if sd > 0 else 1.0)
    if mode == "minmax":
        tmin = float(stats.get("min", 0.0))
        tmax = float(stats.get("max", 1.0))
        rng = max(tmax - tmin, 1e-9)
        return (t_sec - tmin) / rng * 2.0 - 1.0
    raise ValueError(f"unknown time_norm mode: {mode}")


class FeatureBuilder:
    """Utility to accumulate feature tensors with group metadata."""

    def __init__(self) -> None:
        self._parts: List[torch.Tensor] = []
        self._counts: Dict[str, int] = {}
        self._order: List[str] = []

    def _register(self, kind: str, tensor: torch.Tensor) -> None:
        if tensor is None:
            return
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)
        self._parts.append(tensor)
        width = int(tensor.shape[-1])
        self._counts[kind] = self._counts.get(kind, 0) + width
        if kind not in self._order:
            self._order.append(kind)

    def add(self, kind: str, tensor: torch.Tensor) -> None:
        self._register(kind, tensor)

    def add_space(self, tensor: torch.Tensor) -> None:
        self._register("space", tensor)

    def add_time(self, tensor: torch.Tensor) -> None:
        self._register("time", tensor)

    def add_other(self, tensor: torch.Tensor) -> None:
        self._register("other", tensor)

    def build(self) -> torch.Tensor:
        if not self._parts:
            raise ValueError("FeatureBuilder requires at least one tensor before build().")
        if len(self._parts) == 1:
            return self._parts[0]
        return torch.cat(self._parts, dim=-1)

    def layout(self) -> Dict[str, Union[int, List[str]]]:
        full = sum(self._counts.values())
        layout: Dict[str, int | List[str]] = {**self._counts}
        layout["order"] = list(self._order)
        layout["full"] = full
        return layout
