"""Smoothness/spectral metrics for Poseidon models."""

from __future__ import annotations

import torch


def laplacian_penalty(
    lat_deg: torch.Tensor,
    lon_deg: torch.Tensor,
    values: torch.Tensor,
    *,
    max_points: int = 512,
    k: int = 8,
    length_km: float = 50.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Approximate Laplacian smoothness using a local kNN graph on the sphere."""
    lat = lat_deg.reshape(-1)
    lon = lon_deg.reshape(-1)
    vals = values.reshape(-1)
    n = lat.numel()
    device = lat.device
    dtype = lat.dtype
    if n == 0:
        return torch.zeros((), device=device, dtype=dtype)

    if n > max_points > 0:
        idx = torch.randperm(n, device=device)[:max_points]
        lat = lat[idx]
        lon = lon[idx]
        vals = vals[idx]
        n = lat.numel()

    if n <= 1 or k <= 0:
        return torch.zeros((), device=device, dtype=dtype)

    lat_r = torch.deg2rad(lat)
    lon_r = torch.deg2rad(lon)
    cos_lat = torch.cos(lat_r)
    x = cos_lat * torch.cos(lon_r)
    y = cos_lat * torch.sin(lon_r)
    z = torch.sin(lat_r)
    coords = torch.stack((x, y, z), dim=1)
    chord = torch.cdist(coords, coords)
    arc = 2.0 * 6371.0 * torch.asin(torch.clamp(chord * 0.5, max=1.0))

    k_eff = min(k + 1, n)
    dist, idx = torch.topk(arc, k=k_eff, largest=False)
    dist = dist[:, 1:]
    idx = idx[:, 1:]

    neighbor_vals = vals[idx]
    center_vals = vals.unsqueeze(1)
    diff = center_vals - neighbor_vals

    denom = (dist + eps) ** 2
    weights = torch.exp(-(dist / max(length_km, eps)) ** 2)
    penalty = (weights * (diff * diff) / denom).mean()
    return penalty
