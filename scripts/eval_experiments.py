                      
"""Batch evaluation utility for Poseidon experiment folders."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from poseidon.training import LitRegressor, create_datamodule_from_config


@dataclass
class EvaluationResult:
    run_name: str
    checkpoint: Path
    samples: int
    rmse: float
    mae: float
    bias: float
    max_abs: float
    r2: float
    corr: float
    per_pixel_corr_median: float
    per_pixel_corr_mean: float
    per_pixel_corr_count: int
    metrics_path: Path


def _load_config(cfg_path: Path) -> Dict[str, Any]:
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration is not a mapping: {cfg_path}")
    return cfg


def _extract_best_checkpoint(metrics_file: Path) -> Optional[Path]:
    if not metrics_file.exists():
        return None
    with metrics_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("best_model_path="):
                raw = line.split("=", 1)[1].strip()
                if raw:
                    return Path(raw).expanduser().resolve()
                break
    return None


def _fallback_checkpoint(run_dir: Path) -> Optional[Path]:
    ckpt_dir = run_dir.parent / "checkpoints"
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob(f"{run_dir.name}_*.ckpt"))
    if candidates:
        return candidates[-1].resolve()
    return None


def _collect_run_dirs(exp_dir: Path, includes: Sequence[str] | None, excludes: Sequence[str] | None) -> List[Path]:
    run_dirs: List[Path] = []
    for child in sorted(exp_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "checkpoints":
            continue
        if not (child / "config_used.yaml").exists():
            continue
        if includes and not any(child.match(pattern) for pattern in includes):
            continue
        if excludes and any(child.match(pattern) for pattern in excludes):
            continue
        run_dirs.append(child)
    return run_dirs


def _ensure_output_dir(run_dir: Path) -> Path:
    out_dir = run_dir / "evaluation"
    out_dir.mkdir(exist_ok=True)
    return out_dir


def _gather_arrays(lit_model: LitRegressor, datamodule, device: torch.device, limit_batches: Optional[int]) -> Dict[str, np.ndarray]:
    lit_model.eval()
    pred_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []
    time_chunks: List[np.ndarray] = []
    lat_chunks: List[np.ndarray] = []
    lon_chunks: List[np.ndarray] = []
    pass_chunks: List[np.ndarray] = []
    loader = datamodule.test_dataloader()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            lat = batch["lat"].to(device).float()
            lon = batch["lon"].to(device).float()
            t = batch.get("t")
            if t is not None:
                t = t.to(device).float()
            y = batch["y"].to(device).float()

            pred = lit_model.model(lat, lon, t)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = pred.reshape(-1)
            # Model outputs are already in physical SSH units; normalization is applied only in the loss.
            y = y.reshape(-1)

            pred_chunks.append(pred.detach().cpu().numpy())
            target_chunks.append(y.cpu().numpy())

            if t is not None:
                time_chunks.append(t.reshape(-1).cpu().numpy())
            else:
                time_chunks.append(np.full(y.numel(), np.nan, dtype=np.float32))

            lat_chunks.append(lat.reshape(-1).cpu().numpy())
            lon_chunks.append(lon.reshape(-1).cpu().numpy())

            pass_val = batch.get("pass")
            if pass_val is None:
                pass_array = np.full(y.numel(), -1, dtype=np.int32)
            else:
                if isinstance(pass_val, torch.Tensor):
                    pass_val = pass_val.item()
                pass_array = np.full(y.numel(), int(pass_val), dtype=np.int32)
            pass_chunks.append(pass_array)

            if limit_batches is not None and batch_idx + 1 >= limit_batches:
                break
    if not pred_chunks:
        raise RuntimeError("No samples collected; verify test dataset configuration.")
    preds = np.concatenate(pred_chunks)
    targets = np.concatenate(target_chunks)
    times = np.concatenate(time_chunks)
    lats = np.concatenate(lat_chunks)
    lons = np.concatenate(lon_chunks)
    pass_ids = np.concatenate(pass_chunks)
    return {"preds": preds, "targets": targets, "times": times, "lats": lats, "lons": lons, "pass_ids": pass_ids}


def _group_indices_by_pixel(lats: np.ndarray, lons: np.ndarray, decimals: int = 5) -> Dict[tuple[float, float], np.ndarray]:
    if lats.shape != lons.shape:
        raise ValueError("Latitude and longitude arrays must share shape")
    coords = np.stack((np.round(lats, decimals=decimals), np.round(lons, decimals=decimals)), axis=1)
    groups: Dict[tuple[float, float], List[int]] = {}
    for idx, (lat, lon) in enumerate(coords):
        key = (float(lat), float(lon))
        groups.setdefault(key, []).append(idx)
    return {key: np.asarray(idxs, dtype=np.int64) for key, idxs in groups.items()}


def _pixel_stats(
    groups: Dict[tuple[float, float], np.ndarray],
    preds: np.ndarray,
    targets: np.ndarray,
) -> tuple[Dict[str, float], Dict[str, np.ndarray]]:
    corrs: List[float] = []
    lat_list: List[float] = []
    lon_list: List[float] = []
    target_mean_list: List[float] = []
    pred_mean_list: List[float] = []
    residual_mean_list: List[float] = []
    count_list: List[int] = []
    corr_list: List[float] = []

    for (lat_key, lon_key), idxs in groups.items():
        if idxs.size == 0:
            continue
        pix_pred = preds[idxs]
        pix_target = targets[idxs]
        diff = pix_pred - pix_target

        lat_list.append(lat_key)
        lon_list.append(lon_key)
        target_mean_list.append(float(np.mean(pix_target)))
        pred_mean_list.append(float(np.mean(pix_pred)))
        residual_mean_list.append(float(np.mean(diff)))
        count_list.append(int(idxs.size))

        corr_val = float("nan")
        if idxs.size >= 2:
            pred_std = np.std(pix_pred)
            target_std = np.std(pix_target)
            if pred_std > 0.0 and target_std > 0.0:
                corr_val = float(np.corrcoef(pix_pred, pix_target)[0, 1])

        corr_list.append(corr_val)
        if np.isfinite(corr_val):
            corrs.append(corr_val)

    if corrs:
        corr_arr = np.asarray(corrs, dtype=np.float32)
        stats = {
            "per_pixel_corr_median": float(np.median(corr_arr)),
            "per_pixel_corr_mean": float(np.mean(corr_arr)),
            "per_pixel_corr_count": int(corr_arr.size),
        }
    else:
        stats = {
            "per_pixel_corr_median": float("nan"),
            "per_pixel_corr_mean": float("nan"),
            "per_pixel_corr_count": 0,
        }

    aggregated = {
        "lat": np.asarray(lat_list, dtype=np.float32),
        "lon": np.asarray(lon_list, dtype=np.float32),
        "target_mean": np.asarray(target_mean_list, dtype=np.float32),
        "pred_mean": np.asarray(pred_mean_list, dtype=np.float32),
        "residual_mean": np.asarray(residual_mean_list, dtype=np.float32),
        "corr": np.asarray(corr_list, dtype=np.float32),
        "count": np.asarray(count_list, dtype=np.int32),
    }

    return stats, aggregated


def _robust_limits(
    arrays: Sequence[np.ndarray] | np.ndarray,
    lower: float = 2.0,
    upper: float = 98.0,
    symmetric: bool = False,
) -> Optional[tuple[float, float]]:
    if isinstance(arrays, np.ndarray):
        sequences = [arrays]
    else:
        sequences = list(arrays)
    finite_chunks: List[np.ndarray] = []
    for arr in sequences:
        if arr is None:
            continue
        np_arr = np.asarray(arr)
        mask = np.isfinite(np_arr)
        if np.any(mask):
            finite_chunks.append(np_arr[mask])
    if not finite_chunks:
        return None
    data = np.concatenate(finite_chunks)
    if data.size == 0:
        return None
    if symmetric:
        bound = float(np.nanpercentile(np.abs(data), upper))
        if not np.isfinite(bound) or bound <= 0.0:
            return None
        return (-bound, bound)
    low = float(np.nanpercentile(data, lower))
    high = float(np.nanpercentile(data, upper))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return None
    return (low, high)


def _compute_map_limits(pixel_data: Dict[str, np.ndarray]) -> Dict[str, Optional[tuple[float, float]]]:
    target = pixel_data.get("target_mean")
    pred = pixel_data.get("pred_mean")
    residual = pixel_data.get("residual_mean")
    target_pred_limits = _robust_limits([arr for arr in (target, pred) if arr is not None])
    residual_limits = _robust_limits(residual if residual is not None else [], symmetric=True)
    limits = {
        "target": target_pred_limits,
        "pred": target_pred_limits,
        "residual": residual_limits,
    }
    return limits


def _safe_float(value: float) -> float:
    if value is None:
        return float("nan")
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return val


def _format_metric(value: float, fmt: str = ".3f", nan_token: str = "nan") -> str:
    val = _safe_float(value)
    if not np.isfinite(val):
        return nan_token
    return format(val, fmt)


def _sort_key(value: float) -> float:
    val = _safe_float(value)
    if not np.isfinite(val):
        return float("-inf")
    return val


def _compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    diff = preds - targets
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    max_abs = float(np.max(np.abs(diff)))
    denom = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1.0 - np.sum(diff ** 2) / denom) if denom > 0 else float("nan")
    pred_std = float(np.std(preds))
    target_std = float(np.std(targets))
    if pred_std > 0.0 and target_std > 0.0:
        corr = float(np.corrcoef(preds, targets)[0, 1])
    else:
        corr = float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "max_abs": max_abs,
        "r2": r2,
        "corr": corr,
    }


def _downsample(values: np.ndarray, max_points: int) -> np.ndarray:
    if values.size <= max_points:
        return values
    idx = np.linspace(0, values.size - 1, num=max_points, dtype=np.int64)
    return values[idx]


def _format_iso(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return f"{ts:.2f}"


def _parse_time_arg(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    iso_text = text
    if iso_text.endswith("Z"):
        iso_text = iso_text[:-1] + "+00:00"
    dt = datetime.fromisoformat(iso_text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _make_plots(
    out_dir: Path,
    run_name: str,
    preds: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    pass_details: Sequence[Dict[str, Any]],
    pass_plot_time: Optional[float],
    pass_plot_window: float,
    max_points: int,
    max_pass_maps: int,
) -> None:
    plot_dir = out_dir
    plot_dir.mkdir(parents=True, exist_ok=True)

    sample_preds = _downsample(preds, max_points)
    sample_targets = _downsample(targets, max_points)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sample_targets, sample_preds, s=6, alpha=0.5, edgecolor="none")
    lims = [min(sample_targets.min(), sample_preds.min()), max(sample_targets.max(), sample_preds.max())]
    ax.plot(lims, lims, "k--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{run_name} Pred vs Target")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(plot_dir / "pred_vs_target.png", dpi=150)
    plt.close(fig)

    if not pass_details or max_pass_maps <= 0:
        return

    window = max(float(pass_plot_window), 0.0)
    pass_dir_root = plot_dir / "per_pass_plots"
    pass_dir_root.mkdir(parents=True, exist_ok=True)
    for detail in pass_details[:max_pass_maps]:
        raw = detail.get("raw") or {}
        lat_arr = np.asarray(raw.get("lat"), dtype=np.float32)
        lon_arr = np.asarray(raw.get("lon"), dtype=np.float32)
        time_arr = np.asarray(raw.get("time"), dtype=np.float64)
        pred_arr = np.asarray(raw.get("preds"), dtype=np.float32)
        target_arr = np.asarray(raw.get("targets"), dtype=np.float32)
        if lat_arr.size == 0 or lon_arr.size == 0:
            continue

        finite_mask = np.isfinite(time_arr)
        candidate_times = time_arr[finite_mask]
        if candidate_times.size == 0:
            chosen_time = None
        elif pass_plot_time is not None and np.isfinite(pass_plot_time):
            chosen_time = float(pass_plot_time)
        else:
            chosen_time = float(np.median(candidate_times))

        if chosen_time is not None:
            if window > 0.0:
                mask = np.abs(time_arr - chosen_time) <= window
            else:
                idx = int(np.argmin(np.abs(time_arr - chosen_time)))
                mask = np.zeros_like(time_arr, dtype=bool)
                mask[idx] = True
        else:
            mask = np.ones_like(lat_arr, dtype=bool)

        if not np.any(mask):
            continue

        use_lat = lat_arr[mask]
        use_lon = lon_arr[mask]
        use_pred = pred_arr[mask]
        use_target = target_arr[mask]
        use_time = time_arr[mask]
        timestamp_text = _format_iso(float(np.median(use_time))) if use_time.size else "unknown"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        for ax, data, title in zip(
            axes,
            (use_target, use_pred),
            ("Target", "Prediction"),
        ):
            sc = ax.scatter(use_lon, use_lat, c=data, cmap="coolwarm", s=12, edgecolors="none")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(title)
            ax.grid(True, linewidth=0.3, alpha=0.4)
            fig.colorbar(sc, ax=ax, shrink=0.85, label="SSH (m)")
        pass_id = detail.get("pass_id")
        title_bits: List[str] = [run_name]
        if pass_id is not None:
            title_bits.append(f"Pass {pass_id}")
        title_bits.append(timestamp_text)
        fig.suptitle(" | ".join(title_bits))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        pass_dir = pass_dir_root / (f"pass_{int(pass_id):03d}" if pass_id is not None else "pass_unknown")
        pass_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(pass_dir / "timestamp_scatter.png", dpi=150)
        plt.close(fig)


def _plot_spatial_maps(
    map_dir: Path,
    run_name: str,
    pixel_data: Dict[str, np.ndarray],
    limits: Dict[str, Optional[tuple[float, float]]],
    title_suffix: str,
) -> None:
    lat = pixel_data.get("lat")
    lon = pixel_data.get("lon")
    if lat is None or lon is None or lat.size == 0:
        return

    target_mean = pixel_data.get("target_mean")
    pred_mean = pixel_data.get("pred_mean")
    residual_mean = pixel_data.get("residual_mean")
    corr = pixel_data.get("corr")

    map_dir.mkdir(parents=True, exist_ok=True)

    def _scatter(values: np.ndarray, cmap: str, filename: str, label: str, vmin=None, vmax=None) -> None:
        if values is None or values.size == 0:
            return
        data = values.astype(np.float32)
        valid_mask = np.isfinite(data)
        if not np.any(valid_mask):
            return
        data_to_plot = data.copy()
        data_to_plot[~valid_mask] = np.nan
        vmin_local = vmin
        vmax_local = vmax

        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(lon, lat, c=data_to_plot, s=12, cmap=cmap, vmin=vmin_local, vmax=vmax_local, edgecolors="none")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        title_lines = [run_name]
        if title_suffix:
            title_lines.append(title_suffix)
        title_lines.append(label)
        ax.set_title("\n".join(title_lines))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.2, alpha=0.5)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.ax.set_ylabel(label)
        fig.tight_layout()
        fig.savefig(map_dir / filename, dpi=150)
        plt.close(fig)

    target_limits = limits.get("target") if limits else None
    pred_limits = limits.get("pred") if limits else None
    residual_limits = limits.get("residual") if limits else None

    if target_limits is not None:
        target_vmin, target_vmax = target_limits
    else:
        target_vmin = target_vmax = None

    if pred_limits is not None:
        pred_vmin, pred_vmax = pred_limits
    else:
        pred_vmin = pred_vmax = None

    if residual_limits is not None:
        res_vmin, res_vmax = residual_limits
    else:
        res_pair = _robust_limits(residual_mean if residual_mean is not None else [], symmetric=True)
        if res_pair is not None:
            res_vmin, res_vmax = res_pair
        else:
            res_vmin = res_vmax = None

    _scatter(target_mean, "viridis", "target_mean.png", "Per-Pixel Target Mean", vmin=target_vmin, vmax=target_vmax)
    _scatter(pred_mean, "viridis", "prediction_mean.png", "Per-Pixel Prediction Mean", vmin=pred_vmin, vmax=pred_vmax)
    _scatter(residual_mean, "coolwarm", "residual_mean.png", "Per-Pixel Residual (Pred - Target)", vmin=res_vmin, vmax=res_vmax)

    if corr is not None and corr.size > 0:
        _scatter(corr, "coolwarm", "per_pixel_correlation.png", "Per-Pixel Correlation", vmin=-1.0, vmax=1.0)


def _write_metrics(out_dir: Path, metrics: Dict[str, float], samples: int, checkpoint: Path) -> Path:
    payload = {
        "samples": samples,
        "checkpoint": str(checkpoint),
        **metrics,
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return metrics_path


def _write_per_pixel_metrics(out_dir: Path, per_pixel: Dict[str, np.ndarray]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "per_pixel_metrics.npz"
    np.savez_compressed(path, **per_pixel)
    return path


def evaluate_run(
    run_dir: Path,
    device: torch.device,
    limit_batches: Optional[int],
    max_points: int,
    max_pass_maps: int,
    *,
    pass_plot_time: Optional[float],
    pass_plot_window: float,
    produce_plots: bool,
) -> EvaluationResult:
    cfg_path = run_dir / "config_used.yaml"
    metrics_file = run_dir / "final_metrics.txt"
    cfg = _load_config(cfg_path)

    checkpoint = _extract_best_checkpoint(metrics_file)
    if checkpoint is None:
        checkpoint = _fallback_checkpoint(run_dir)
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found for run {run_dir}")

    datamodule = create_datamodule_from_config(cfg)
    datamodule.setup("fit")
    datamodule.setup("test")

    lit_model = LitRegressor.load_from_checkpoint(
        str(checkpoint),
        cfg=cfg,
        dm_stats=datamodule.stats,
        strict=False,
    )
    lit_model.to(device)

    arrays = _gather_arrays(lit_model, datamodule, device, limit_batches)
    preds = arrays["preds"]
    targets = arrays["targets"]
    times = arrays["times"]
    lats = arrays["lats"]
    lons = arrays["lons"]
    pass_ids = arrays.get("pass_ids")
    if pass_ids is None:
        pass_ids = np.full(preds.shape, -1, dtype=np.int32)

    metrics = _compute_metrics(preds, targets)
    pixel_groups = _group_indices_by_pixel(lats, lons)
    pixel_metrics, pixel_data = _pixel_stats(pixel_groups, preds, targets)
    metrics.update(pixel_metrics)
    out_dir = _ensure_output_dir(run_dir)
    per_pixel_path = _write_per_pixel_metrics(out_dir, pixel_data)
    metrics["per_pixel_metrics_file"] = str(per_pixel_path)

    map_limits = _compute_map_limits(pixel_data)
    if produce_plots:
        map_suffix = (
            f"| R^2={_format_metric(metrics['r2'], '.3f')} "
            f"| corr={_format_metric(metrics['corr'], '.3f')} | N={preds.size:,}"
        )
        _plot_spatial_maps(out_dir / "per_pixel_maps", run_dir.name, pixel_data, map_limits, map_suffix)

    pass_details: List[Dict[str, Any]] = []
    per_pass_info: Dict[str, Any] = {}
    unique_passes = np.unique(pass_ids)
    unique_passes = unique_passes[np.isfinite(unique_passes)]
    unique_passes = unique_passes.astype(int, copy=False)
    unique_passes = unique_passes[unique_passes >= 0]
    if unique_passes.size:
        unique_passes.sort()
    if unique_passes.size:
        for pid in unique_passes:
            mask = pass_ids == pid
            if not np.any(mask):
                continue
            pass_preds = preds[mask]
            pass_targets = targets[mask]
            pass_metrics = _compute_metrics(pass_preds, pass_targets)
            pass_groups = _group_indices_by_pixel(lats[mask], lons[mask])
            pass_pixel_stats, pass_pixel_data = _pixel_stats(pass_groups, pass_preds, pass_targets)
            pass_dir = out_dir / "per_pass" / f"pass_{pid:03d}"
            pass_metrics_path = _write_per_pixel_metrics(pass_dir, pass_pixel_data)
            pass_map_limits = _compute_map_limits(pass_pixel_data)
            sample_count = int(pass_preds.size)
            if produce_plots:
                pass_suffix = (
                    f"| R^2={_format_metric(pass_metrics['r2'], '.3f')} "
                    f"| corr={_format_metric(pass_metrics['corr'], '.3f')} | N={sample_count:,}"
                )
                _plot_spatial_maps(
                    pass_dir / "per_pixel_maps",
                    f"{run_dir.name} Pass {pid:03d}",
                    pass_pixel_data,
                    pass_map_limits,
                    pass_suffix,
                )
            pass_record = {
                "pass_id": int(pid),
                "metrics": pass_metrics,
                "pixel_stats": pass_pixel_stats,
                "pixel_data": pass_pixel_data,
                "samples": sample_count,
                "per_pixel_metrics_file": str(pass_metrics_path),
                "raw": {
                    "lat": lats[mask],
                    "lon": lons[mask],
                    "time": times[mask],
                    "preds": pass_preds,
                    "targets": pass_targets,
                },
            }
            pass_details.append(pass_record)
            per_pass_info[str(int(pid))] = {
                "samples": sample_count,
                "rmse": pass_metrics["rmse"],
                "mae": pass_metrics["mae"],
                "bias": pass_metrics["bias"],
                "max_abs": pass_metrics["max_abs"],
                "r2": pass_metrics["r2"],
                "corr": pass_metrics["corr"],
                "per_pixel_corr_median": pass_pixel_stats["per_pixel_corr_median"],
                "per_pixel_corr_mean": pass_pixel_stats["per_pixel_corr_mean"],
                "per_pixel_corr_count": pass_pixel_stats["per_pixel_corr_count"],
                "per_pixel_metrics_file": str(pass_metrics_path),
            }

    if pass_details:
        pass_details.sort(
            key=lambda d: (_sort_key(d["metrics"].get("r2")), _sort_key(d["metrics"].get("corr")), d.get("samples", 0)),
            reverse=True,
        )
    if per_pass_info:
        metrics["per_pass"] = per_pass_info

    if produce_plots:
        _make_plots(
            out_dir,
            run_dir.name,
            preds,
            targets,
            times,
            pass_details,
            pass_plot_time=pass_plot_time,
            pass_plot_window=pass_plot_window,
            max_points=max_points,
            max_pass_maps=max_pass_maps,
        )
    metrics_path = _write_metrics(out_dir, metrics, preds.size, checkpoint)

    return EvaluationResult(
        run_name=run_dir.name,
        checkpoint=checkpoint,
        samples=int(preds.size),
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        bias=metrics["bias"],
        max_abs=metrics["max_abs"],
        r2=metrics["r2"],
        corr=metrics["corr"],
        per_pixel_corr_median=pixel_metrics["per_pixel_corr_median"],
        per_pixel_corr_mean=pixel_metrics["per_pixel_corr_mean"],
        per_pixel_corr_count=pixel_metrics["per_pixel_corr_count"],
        metrics_path=metrics_path,
    )


def _write_summary(exp_dir: Path, results: Sequence[EvaluationResult]) -> None:
    if not results:
        return
    summary_path = exp_dir / "evaluation_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "run",
            "checkpoint",
            "samples",
            "rmse",
            "mae",
            "bias",
            "max_abs",
            "r2",
            "corr",
            "per_pixel_corr_median",
            "per_pixel_corr_mean",
            "per_pixel_corr_count",
            "metrics_file",
        ])
        for item in results:
            median_val = item.per_pixel_corr_median
            mean_val = item.per_pixel_corr_mean
            writer.writerow([
                item.run_name,
                str(item.checkpoint),
                item.samples,
                f"{item.rmse:.6f}",
                f"{item.mae:.6f}",
                f"{item.bias:.6f}",
                f"{item.max_abs:.6f}",
                f"{item.r2:.6f}",
                f"{item.corr:.6f}",
                f"{median_val:.6f}" if np.isfinite(median_val) else "nan",
                f"{mean_val:.6f}" if np.isfinite(mean_val) else "nan",
                str(item.per_pixel_corr_count),
                str(item.metrics_path),
            ])
    print(f"Wrote summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a folder of Poseidon experiment runs.")
    parser.add_argument("--experiment-dir", required=True, help="Directory containing run subfolders.")
    parser.add_argument("--include", nargs="*", default=None, help="Glob patterns to include (e.g. W*_D3).")
    parser.add_argument("--exclude", nargs="*", default=None, help="Glob patterns to skip.")
    parser.add_argument("--device", default="cpu", help="Device for inference (cpu, mps, cuda:0, ...).")
    parser.add_argument("--limit-batches", type=int, default=None, help="Optional cap on number of test batches per run.")
    parser.add_argument("--max-points", type=int, default=20000, help="Max points for scatter plots.")
    parser.add_argument("--max-pass-maps", type=int, default=2, help="Maximum number of test passes to include in per-pass plots.")
    parser.add_argument(
        "--pass-plot-time",
        help="Timestamp (seconds or ISO8601) to center per-pass plots. Defaults to the median pass time when omitted.",
    )
    parser.add_argument(
        "--pass-plot-window",
        type=float,
        default=0.0,
        help="Half-width (seconds) around the selected timestamp when plotting per-pass samples (default 0 = single timestamp).",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Disable plot generation during evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.experiment_dir).expanduser().resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    run_dirs = _collect_run_dirs(exp_dir, args.include, args.exclude)
    if not run_dirs:
        raise RuntimeError(f"No run directories found under {exp_dir}")

    device = torch.device(args.device)
    results: List[EvaluationResult] = []
    for run_dir in run_dirs:
        print(f"Evaluating {run_dir.name}...")
        result = evaluate_run(
            run_dir,
            device=device,
            limit_batches=args.limit_batches,
            max_points=args.max_points,
            max_pass_maps=args.max_pass_maps,
            pass_plot_time=_parse_time_arg(args.pass_plot_time),
            pass_plot_window=args.pass_plot_window,
            produce_plots=not args.skip_plots,
        )
        print(
            f"  rmse={result.rmse:.4f} mae={result.mae:.4f} bias={result.bias:.4f} "
            f"corr={result.corr:.4f} pix_corr_med={result.per_pixel_corr_median:.4f} "
            f"samples={result.samples} checkpoint={result.checkpoint.name}"
        )
        results.append(result)

    _write_summary(exp_dir, results)


if __name__ == "__main__":
    main()
