from __future__ import annotations

from typing import Literal, Mapping, Any, Sequence

import torch

__all__ = [
    "distance_to_coast_weight",
    "cross_track_distance_weight",
    "normalize_weights",
    "prepare_loss_weight_config",
    "compute_loss_weights",
]

_Units = Literal["km", "m"]
_NormalizeKind = Literal["none", "mean", "sum", "max"]
_STAGE_ALIASES: dict[str, str] = {
    "train": "train",
    "training": "train",
    "fit": "train",
    "validate": "validate",
    "validation": "validate",
    "val": "validate",
    "eval": "validate",
    "evaluation": "validate",
    "test": "test",
    "testing": "test",
    "predict": "predict",
    "inference": "predict",
}


def _canonical_stage(name: str) -> str:
    return _STAGE_ALIASES.get(name.lower(), name.lower())


def _to_tensor(values: torch.Tensor | float | int) -> torch.Tensor:
    if torch.is_tensor(values):
        return values
    return torch.as_tensor(values, dtype=torch.get_default_dtype())


def _to_kilometers(distance: torch.Tensor | float | int, units: _Units) -> torch.Tensor:
    distance_tensor = _to_tensor(distance)
    if units == "km":
        return distance_tensor
    if units == "m":
        return distance_tensor / 1000.0
    raise ValueError(f"Unsupported distance units '{units}'. Use 'km' or 'm'.")


def normalize_weights(weights: torch.Tensor, mode: _NormalizeKind = "mean", eps: float = 1e-6) -> torch.Tensor:
    """Rescale weights to keep training numerics stable."""
    if mode == "none":
        return weights
    if mode == "mean":
        denom = torch.clamp(weights.mean(), min=eps)
        return weights / denom
    if mode == "sum":
        denom = torch.clamp(weights.sum(), min=eps)
        return weights / denom
    if mode == "max":
        denom = torch.clamp(weights.max(), min=eps)
        return weights / denom
    raise ValueError(f"Unsupported normalization mode '{mode}'.")


def distance_to_coast_weight(
    distance: torch.Tensor | float | int,
    *,
    units: _Units = "km",
    method: Literal["linear", "exp"] = "linear",
    min_km: float = 0.0,
    max_km: float | None = 200.0,
    scale_km: float = 50.0,
    invert: bool = True,
    clamp_min: float = 1e-6,
) -> torch.Tensor:
    """
    Convert distance-to-coast values to per-sample weights.

    Parameters
    ----------
    distance: Tensor or numeric
        The distance-to-coast values.
    units: str
        Either "km" or "m".
    method: str
        "linear" uses a clipped ramp between min_km and max_km.
        "exp" applies an exponential decay controlled by scale_km.
    min_km, max_km:
        Bounds applied before scaling. max_km is required for the linear method.
    scale_km:
        Characteristic decay length used by the exponential method.
    invert: bool
        When True (default) near-coast samples receive larger weights.
    clamp_min: float
        Lower bound applied to the final weights.
    """

    dist_km = _to_kilometers(distance, units).clamp_min(min_km)

    if method == "linear":
        if max_km is None or max_km <= min_km:
            raise ValueError("linear weighting requires max_km > min_km")
        span = max(max_km - min_km, 1e-6)
        norm = (dist_km - min_km) / span
        norm = norm.clamp_(0.0, 1.0)
        weight = 1.0 - norm if invert else norm
    elif method == "exp":
        scale = max(scale_km, 1e-6)
        base = torch.exp(-((dist_km - min_km).clamp_min(0.0) / scale))
        weight = base if invert else (1.0 - base)
    else:
        raise ValueError(f"Unsupported method '{method}'.")

    return weight.clamp_min(clamp_min)


def cross_track_distance_weight(
    distance: torch.Tensor | float | int,
    *,
    units: _Units = "km",
    method: Literal["gaussian", "linear"] = "gaussian",
    sigma_km: float = 10.0,
    clamp_min: float = 1e-6,
) -> torch.Tensor:
    """
    Derive weights from cross-track distance (absolute offset from the nadir)."""

    dist_km = _to_kilometers(distance, units).abs()

    if method == "gaussian":
        sigma = max(sigma_km, 1e-6)
        weight = torch.exp(-0.5 * (dist_km / sigma) ** 2)
    elif method == "linear":
        reach = max(sigma_km, 1e-6)
        weight = torch.clamp(1.0 - dist_km / reach, min=0.0)
    else:
        raise ValueError(f"Unsupported method '{method}'.")

    return weight.clamp_min(clamp_min)


_COMPONENTS: dict[str, dict[str, Any]] = {
    "distance_to_coast": {
        "tensor_key": "distance_to_coast",
        "fn": distance_to_coast_weight,
        "allowed": {
            "units",
            "method",
            "min_km",
            "max_km",
            "scale_km",
            "invert",
            "clamp_min",
        },
    },
    "cross_track_distance": {
        "tensor_key": "cross_track_distance",
        "fn": cross_track_distance_weight,
        "allowed": {
            "units",
            "method",
            "sigma_km",
            "clamp_min",
        },
    },
}


def _as_set(values: Sequence[str] | str | None, default: Sequence[str]) -> set[str]:
    if values is None:
        items = default
    elif isinstance(values, str):
        items = [values]
    else:
        items = values
    return {_canonical_stage(str(v)) for v in items}


def prepare_loss_weight_config(cfg: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(cfg, Mapping):
        return None

    components: list[dict[str, Any]] = []
    for name, spec in _COMPONENTS.items():
        opts = cfg.get(name)
        if not isinstance(opts, Mapping):
            continue
        params = {}
        for key in spec["allowed"]:
            if key in opts:
                params[key] = opts[key]
        component = {
            "name": name,
            "tensor_key": spec["tensor_key"],
            "fn": spec["fn"],
            "params": params,
            "normalize": opts.get("normalize"),
            "normalize_eps": float(opts.get("normalize_eps", cfg.get("normalize_eps", 1e-6))),
            "scale": float(opts.get("scale", 1.0)),
        }
        components.append(component)

    if not components:
        return None

    combine = str(cfg.get("combine", "product")).lower()
    if combine not in {"product", "multiply"}:
        raise ValueError(f"Unsupported combine mode '{combine}'. Use 'product'.")

    prepared = {
        "components": components,
        "combine": "product",
        "normalize": cfg.get("normalize"),
        "normalize_eps": float(cfg.get("normalize_eps", 1e-6)),
        "apply_to": _as_set(cfg.get("apply_to"), ["train"]),
    }
    return prepared


def compute_loss_weights(
    prepared_cfg: dict[str, Any] | None,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    stage: str = "train",
) -> torch.Tensor | None:
    if not prepared_cfg:
        return None

    stage_name = _canonical_stage(stage)
    if stage_name not in prepared_cfg.get("apply_to", {"train"}):
        return None

    weights: torch.Tensor | None = None

    for component in prepared_cfg["components"]:
        tensor = batch.get(component["tensor_key"])
        if tensor is None:
            continue
        values = torch.as_tensor(tensor, dtype=torch.float32, device=device).reshape(-1)
        comp_weight = component["fn"](values, **component["params"])
        if component["normalize"]:
            comp_weight = normalize_weights(
                comp_weight,
                mode=str(component["normalize"]),
                eps=float(component["normalize_eps"]),
            )
        scale = float(component.get("scale", 1.0))
        if scale != 1.0:
            comp_weight = comp_weight * scale
        comp_weight = comp_weight.reshape(-1)
        weights = comp_weight if weights is None else weights * comp_weight

    if weights is None:
        return None

    global_norm = prepared_cfg.get("normalize")
    if global_norm:
        weights = normalize_weights(
            weights,
            mode=str(global_norm),
            eps=float(prepared_cfg.get("normalize_eps", 1e-6)),
        )

    return weights.reshape(-1)
