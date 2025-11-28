from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class TargetNormalizer:
    """Simple affine normalizer supporting identity and z-score scaling."""

    kind: str = "none"
    mean: float = 0.0
    std: float = 1.0
    eps: float = 1e-6

    @property
    def active(self) -> bool:
        return self.kind not in {"none", "identity"}

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return values
        scale = torch.as_tensor(max(self.std, self.eps), dtype=values.dtype, device=values.device)
        shift = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        return (values - shift) / scale

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return values
        scale = torch.as_tensor(max(self.std, self.eps), dtype=values.dtype, device=values.device)
        shift = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        return values * scale + shift

    def to_dict(self) -> Dict[str, float]:
        payload: Dict[str, float] = {"type": self.kind}
        if self.active:
            payload.update({"mean": self.mean, "std": self.std, "eps": self.eps})
        return payload

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "TargetNormalizer":
        if not data:
            return cls(kind="none")
        kind = str(data.get("type", "none")).lower()
        if kind in {"none", "identity"}:
            return cls(kind="none")
        if kind in {"zscore", "standard", "standardize", "standardization"}:
            mean = float(data.get("mean", 0.0))
            std = float(data.get("std", 1.0))
            eps = float(data.get("eps", 1e-6))
            return cls(kind="zscore", mean=mean, std=max(std, eps), eps=eps)
        raise ValueError(f"Unknown normalizer type: {kind}")

    @classmethod
    def fit(cls, kind: str, values: np.ndarray, *, eps: float = 1e-6) -> "TargetNormalizer":
        kind = str(kind).lower()
        if kind in {"none", "identity"}:
            return cls(kind="none")
        if kind in {"zscore", "standard", "standardize", "standardization"}:
            if values.size == 0:
                return cls(kind="zscore", mean=0.0, std=1.0, eps=eps)
            mean = float(values.mean())
            std = float(values.std())
            return cls(kind="zscore", mean=mean, std=max(std, eps), eps=eps)
        raise ValueError(f"Unknown normalizer type: {kind}")


class TimeJulian:
    """Convert timestamps to Julian days, matching the legacy preprocessing."""

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        values = sample["temporal"]
        shape = values.shape
        julian = pd.to_datetime(values.reshape(-1)).to_julian_date().to_numpy()
        sample["temporal"] = julian.reshape(shape)
        return sample


class TimeJulianMinMax:
    """Julian conversion followed by global min-max normalization."""

    def __init__(self, time_min: str = "2005-01-10", time_max: str = "2022-01-01") -> None:
        self.time_min = pd.to_datetime(np.datetime64(time_min)).to_julian_date()
        self.time_max = pd.to_datetime(np.datetime64(time_max)).to_julian_date()

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        values = sample["temporal"]
        shape = values.shape
        julian = pd.to_datetime(values.reshape(-1)).to_julian_date().to_numpy()
        scaled = (julian - self.time_min) / max(self.time_max - self.time_min, 1e-12)
        sample["temporal"] = scaled.reshape(shape)
        return sample


class TimeMinMax:
    """Direct min-max scaling on datetime64 timestamps."""

    def __init__(self, time_min: str = "2005-01-10", time_max: str = "2022-01-01") -> None:
        self.time_min = np.datetime64(time_min)
        self.time_max = np.datetime64(time_max)

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        values = sample["temporal"].astype("datetime64[ns]")
        scaled = (values - self.time_min) / (self.time_max - self.time_min)
        sample["temporal"] = scaled.astype(np.float32)
        return sample


class ToTensor:
    """Cast processed arrays into float32 tensors."""

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        sample["spatial"] = torch.as_tensor(sample["spatial"], dtype=torch.float32)
        sample["temporal"] = torch.as_tensor(sample["temporal"], dtype=torch.float32)
        if "output" in sample:
            try:
                sample["output"] = torch.as_tensor(sample["output"], dtype=torch.float32)
            except Exception:
                pass
        return sample


def _get_cfg_value(config: Union[Optional[Mapping[str, Any]], Any], key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def transform_factory(config: Optional[Mapping[str, Any]] | Any) -> Sequence[Any]:
    if config is None:
        raise ValueError("transform config must be provided")

    transform_key = _get_cfg_value(config, "time_transform")
    if transform_key is None:
        raise ValueError("time_transform must be specified")

    transforms: list[Any] = []
    kind = str(transform_key).lower()
    if kind == "julian":
        transforms.append(TimeJulian())
    elif kind == "julian_minmax":
        transforms.append(
            TimeJulianMinMax(
                time_min=_get_cfg_value(config, "time_min", "2005-01-10"),
                time_max=_get_cfg_value(config, "time_max", "2022-01-01"),
            )
        )
    elif kind == "minmax":
        transforms.append(
            TimeMinMax(
                time_min=_get_cfg_value(config, "time_min", "2005-01-10"),
                time_max=_get_cfg_value(config, "time_max", "2022-01-01"),
            )
        )
    else:
        raise ValueError(f"Unrecognized transform: {transform_key}")

    transforms.append(ToTensor())
    return transforms


__all__ = [
    "TargetNormalizer",
    "TimeJulian",
    "TimeJulianMinMax",
    "TimeMinMax",
    "ToTensor",
    "transform_factory",
]
