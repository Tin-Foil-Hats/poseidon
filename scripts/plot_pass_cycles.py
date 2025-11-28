                      
"""Plot model predictions for a single SWOT pass across all available cycles."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, List, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from poseidon.data.schema import load_shard
from poseidon.data.shards import load_index
from poseidon.training import LitRegressor, create_datamodule_from_config


def _get(cfg: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _set(cfg: dict[str, Any], path: str, value: Any) -> None:
    cur = cfg
    keys = path.split(".")
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _resolve_path(path: str, *, base: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def _apply_base_paths(cfg: dict[str, Any], *, config_dir: Path) -> None:
    for field in ("data.shards_dir", "data.train_batchshard_dir"):
        value = _get(cfg, field)
        if isinstance(value, str):
            _set(cfg, field, str(_resolve_path(value, base=config_dir)))


def _load_cfg(cfg_path: Path) -> dict[str, Any]:
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, Mapping):
        raise ValueError("Configuration must be a mapping")
    cfg_dict = dict(cfg)
    _apply_base_paths(cfg_dict, config_dir=cfg_path.parent)
    return cfg_dict


def _resolve_checkpoint(cfg_path: Path, checkpoint: str | None, metrics_path: str | None) -> Path:
    if checkpoint:
        return Path(checkpoint).expanduser().resolve()

    candidate = Path(metrics_path).expanduser() if metrics_path else cfg_path.parent / "final_metrics.txt"
    candidate = candidate.resolve()
    if candidate.exists():
        best_path: Path | None = None
        with candidate.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("best_model_path="):
                    raw = line.split("=", 1)[1].strip()
                    if raw:
                        best_path = Path(raw).expanduser()
                    break
        if best_path is not None:
            resolved = best_path.resolve()
            if resolved.exists():
                print(f"Using checkpoint from metrics file: {resolved}")
                return resolved
        print(f"best_model_path missing or invalid in metrics file: {candidate}")
    else:
        print(f"Metrics file not found: {candidate}")

    ckpt_dir = (cfg_path.parent / "checkpoints").resolve()
    if ckpt_dir.is_dir():
        ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            print(f"Using most recent checkpoint from {ckpt_dir}: {ckpts[0]}")
            return ckpts[0].resolve()

    raise FileNotFoundError(
        "Could not locate a checkpoint. Provide --checkpoint explicitly or ensure final_metrics.txt exists."
    )


def _load_model(cfg: dict[str, Any], ckpt_path: Path, device: torch.device):
    datamodule = create_datamodule_from_config(cfg)
    datamodule.setup("fit")
    datamodule.setup("test")
    lit_model = LitRegressor.load_from_checkpoint(str(ckpt_path), cfg=cfg, dm_stats=datamodule.stats)
    lit_model.eval()
    lit_model.to(device)
    return lit_model, datamodule


def _auto_pass(index: List[dict[str, Any]]) -> int:
    tallies: dict[int, int] = {}
    for item in index:
        pas = int(item.get("pass"))
        length = int(item.get("length", 0))
        tallies[pas] = tallies.get(pas, 0) + length
    if not tallies:
        raise RuntimeError("Shard index is empty")
    return max(tallies.items(), key=lambda kv: kv[1])[0]


def _filter_cycles(entries: List[dict[str, Any]], cycles: set[int] | None, max_cycles: int | None) -> List[dict[str, Any]]:
    if cycles:
        entries = [e for e in entries if int(e.get("cycle")) in cycles]
    if not entries:
        return []
    entries.sort(key=lambda e: int(e.get("cycle")))
    if max_cycles is not None and len(entries) > max_cycles:
        subset = sorted(entries, key=lambda e: int(e.get("length", 0)), reverse=True)[:max_cycles]
        subset.sort(key=lambda e: int(e.get("cycle")))
        return subset
    return entries


def _infer_cycle(lit_model: LitRegressor, entry: dict[str, Any], device: torch.device):
    arrays = load_shard(entry["path"])
    lat = np.asarray(arrays["lat"], dtype=np.float32)
    lon = np.asarray(arrays["lon"], dtype=np.float32)
    t = np.asarray(arrays["t"], dtype=np.float32)
    y = np.asarray(arrays["y"], dtype=np.float32)
    mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(y)
    finite_t = np.isfinite(t)
    use_time = bool(finite_t.any())
    if use_time:
        mask &= finite_t
    if mask.sum() == 0:
        raise RuntimeError(f"No finite samples in shard {entry['path']}")
    lat = lat[mask]
    lon = lon[mask]
    y = y[mask]
    if use_time:
        t = t[mask]
    else:
        t = np.zeros_like(y, dtype=np.float32)

    with torch.no_grad():
        lat_t = torch.from_numpy(lat).to(device=device, dtype=torch.float32)
        lon_t = torch.from_numpy(lon).to(device=device, dtype=torch.float32)
        t_t = torch.from_numpy(t).to(device=device, dtype=torch.float32) if use_time and t.size else None
        pred = lit_model.model(lat_t, lon_t, t_t)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = pred.reshape(-1).cpu().numpy()

    if use_time and np.all(np.isfinite(t)) and np.ptp(t) > 0:
        order = np.argsort(t)
        x = (t[order] - t[order][0]) / 60.0
        x_label = "Minutes since pass start"
    else:
        order = np.arange(y.size)
        x = order.astype(np.float32)
        x_label = "Sample index"

    data = {
        "cycle": int(entry.get("cycle")),
        "lat": lat[order],
        "lon": lon[order],
        "time": t[order],
        "truth": y[order],
        "pred": pred[order],
        "x": x.astype(np.float32),
        "x_label": x_label,
    }
    return data


def _scatter_limits(values: List[np.ndarray], symmetric: bool = True) -> tuple[float, float]:
    combined = np.concatenate([v.reshape(-1) for v in values if v.size > 0])
    if combined.size == 0:
        return -1.0, 1.0
    q_low = float(np.percentile(combined, 2))
    q_high = float(np.percentile(combined, 98))
    if symmetric:
        hi = max(abs(q_low), abs(q_high), 1e-6)
        return -hi, hi
    low = min(q_low, q_high)
    high = max(q_low, q_high, 1e-6)
    if low == high:
        high = low + 1e-6
    return low, high


def _plot_series(data: List[dict[str, Any]], pass_id: int, out_path: Path) -> None:
    if not data:
        raise RuntimeError("No cycle data to plot")
    cols = min(3, len(data))
    rows = math.ceil(len(data) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.8, rows * 3.4), squeeze=False, sharex=False, sharey=False)
    axes_flat = axes.ravel()
    for ax, segment in zip(axes_flat, data):
        ax.plot(segment["x"], segment["truth"], label="Truth", linewidth=1.2)
        ax.plot(segment["x"], segment["pred"], label="Pred", linewidth=1.2)
        rmse = float(np.sqrt(np.mean((segment["pred"] - segment["truth"]) ** 2)))
        ax.set_title(f"Cycle {segment['cycle']:03d}  RMSE={rmse:.3f} m  N={segment['truth'].size}")
        ax.set_xlabel(segment["x_label"])
        ax.set_ylabel("SSH (m)")
        ax.grid(True, linewidth=0.3, alpha=0.3)
    for ax in axes_flat[len(data):]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"Pass {pass_id:03d}: predictions vs truth by cycle", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_residuals(data: List[dict[str, Any]], pass_id: int, out_path: Path) -> None:
    cols = min(3, len(data))
    rows = math.ceil(len(data) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.8, rows * 3.4), squeeze=False, sharex=False, sharey=False)
    axes_flat = axes.ravel()
    for ax, segment in zip(axes_flat, data):
        residual = segment["pred"] - segment["truth"]
        ax.plot(segment["x"], residual, label="Residual", linewidth=1.1, color="tab:red")
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        mae = float(np.mean(np.abs(residual)))
        ax.set_title(f"Cycle {segment['cycle']:03d}  MAE={mae:.3f} m")
        ax.set_xlabel(segment["x_label"])
        ax.set_ylabel("ΔSSH (m)")
        ax.grid(True, linewidth=0.3, alpha=0.3)
    for ax in axes_flat[len(data):]:
        ax.axis("off")
    fig.suptitle(f"Pass {pass_id:03d}: residuals by cycle", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_scatter(
    data: List[dict[str, Any]],
    pass_id: int,
    out_path: Path,
    *,
    value_key: str,
    title_suffix: str,
    cmap: str,
    symmetric: bool,
) -> None:
    if not data:
        raise RuntimeError("No cycle data to plot")
    values = [segment[value_key] for segment in data]
    has_negative = any(np.nanmin(v) < 0 for v in values if v.size > 0)
    vmin, vmax = _scatter_limits(values, symmetric=symmetric and has_negative)
    cols = min(3, len(data))
    rows = math.ceil(len(data) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.8, rows * 3.4), squeeze=False, sharex=False, sharey=False)
    axes_flat = axes.ravel()
    for ax, segment in zip(axes_flat, data):
        sc = ax.scatter(
            segment["lon"],
            segment["lat"],
            c=segment[value_key],
            s=8,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Cycle {segment['cycle']:03d}  N={segment['truth'].size}")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.3)
    for ax in axes_flat[len(data):]:
        ax.axis("off")
    fig.colorbar(sc, ax=axes_flat[: len(data)], orientation="horizontal", fraction=0.05, pad=0.08, label="SSH (m)")
    fig.suptitle(f"Pass {pass_id:03d}: {title_suffix}", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot predictions across all cycles for a single SWOT pass.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML (prefer config_used.yaml).")
    parser.add_argument(
        "--checkpoint",
        help=(
            "Lightning checkpoint to load. When omitted, the script searches final_metrics.txt for best_model_path "
            "and otherwise falls back to the newest file under checkpoints/."
        ),
    )
    parser.add_argument("--metrics", help="Optional metrics file with best_model_path entry.")
    parser.add_argument("--pass-id", type=int, help="SWOT pass identifier (e.g. 138). Defaults to densest pass.")
    parser.add_argument("--cycles", type=int, nargs="+", help="Cycle ids to include (e.g. 3 4 5).")
    parser.add_argument("--max-cycles", type=int, help="Limit number of cycles to plot (largest shards kept).")
    parser.add_argument("--output-dir", default="pass_cycle_plots", help="Directory for output figures.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu, mps, cuda:0.")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = _load_cfg(cfg_path)
    ckpt_path = _resolve_checkpoint(cfg_path, args.checkpoint, args.metrics)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(args.device)
    lit_model, _ = _load_model(cfg, ckpt_path, device)

    shards_dir = Path(_get(cfg, "data.shards_dir"))
    if shards_dir is None:
        raise ValueError("data.shards_dir missing in config")
    index = load_index(shards_dir)

    pass_id = args.pass_id if args.pass_id is not None else _auto_pass(index)
    if args.pass_id is None:
        print(f"No pass specified. Using pass {pass_id:03d} with most samples.")

    pass_entries = [dict(item) for item in index if int(item.get("pass")) == pass_id]
    for item in pass_entries:
        item["path"] = str(Path(item["path"]).expanduser().resolve())
    cycles = set(args.cycles) if args.cycles else None
    pass_entries = _filter_cycles(pass_entries, cycles, args.max_cycles)
    if not pass_entries:
        raise RuntimeError("No shards matched the requested pass/cycles")

    cycle_data: List[dict[str, Any]] = []
    for entry in pass_entries:
        data = _infer_cycle(lit_model, entry, device)
        cycle_data.append(data)
        residual = data["pred"] - data["truth"]
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        mae = float(np.mean(np.abs(residual)))
        print(
            f"Cycle {data['cycle']:03d}: N={data['truth'].size:5d}  RMSE={rmse:.4f} m  MAE={mae:.4f} m"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    series_path = output_dir / f"pass_{pass_id:03d}_pred_truth.png"
    residual_path = output_dir / f"pass_{pass_id:03d}_residuals.png"
    scatter_pred_path = output_dir / f"pass_{pass_id:03d}_predicted_scatter.png"
    scatter_truth_path = output_dir / f"pass_{pass_id:03d}_truth_scatter.png"
    scatter_resid_path = output_dir / f"pass_{pass_id:03d}_residual_scatter.png"

    residual_segments: List[dict[str, Any]] = []
    for segment in cycle_data:
        residual = segment["pred"] - segment["truth"]
        residual_segments.append(dict(segment, residual=residual))

    _plot_series(cycle_data, pass_id, series_path)
    _plot_residuals(cycle_data, pass_id, residual_path)
    _plot_scatter(
        cycle_data,
        pass_id,
        scatter_pred_path,
        value_key="pred",
        title_suffix="Predicted SSH (scatter)",
        cmap="viridis",
        symmetric=True,
    )
    _plot_scatter(
        cycle_data,
        pass_id,
        scatter_truth_path,
        value_key="truth",
        title_suffix="Ground Truth SSH (scatter)",
        cmap="viridis",
        symmetric=True,
    )
    _plot_scatter(
        residual_segments,
        pass_id,
        scatter_resid_path,
        value_key="residual",
        title_suffix="Residuals (Pred - Truth)",
        cmap="seismic",
        symmetric=True,
    )

    print(f"Saved prediction series figure to {series_path}")
    print(f"Saved residual series figure to {residual_path}")
    print(f"Saved predicted scatter figure to {scatter_pred_path}")
    print(f"Saved truth scatter figure to {scatter_truth_path}")
    print(f"Saved residual scatter figure to {scatter_resid_path}")


if __name__ == "__main__":
    main()
