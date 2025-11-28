                      
"""Plot predictions vs ground truth for a single SWOT test pass."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

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
    data_fields = (
        "data.shards_dir",
        "data.train_batchshard_dir",
    )
    for field in data_fields:
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


def _load_model(cfg: dict[str, Any], ckpt_path: Path, device: torch.device):
    datamodule = create_datamodule_from_config(cfg)
    datamodule.setup("fit")
    datamodule.setup("test")

    lit_model = LitRegressor.load_from_checkpoint(
        str(ckpt_path),
        cfg=cfg,
        dm_stats=datamodule.stats,
        strict=False,
    )
    lit_model.eval()
    lit_model.to(device)
    return lit_model, datamodule


def _pick_pass(datamodule, pass_index: int):
    test_ds = datamodule.test_ds
    if test_ds is None:
        raise RuntimeError("Datamodule has no test dataset; ensure test_full_pass=true in config")
    if pass_index < 0 or pass_index >= len(test_ds):
        raise IndexError(f"pass_index {pass_index} out of range (0..{len(test_ds) - 1})")
    sample = test_ds[pass_index]
    return sample


def _predict_pass(lit_model: LitRegressor, sample: Mapping[str, torch.Tensor]):
    device = next(lit_model.model.parameters()).device
    with torch.no_grad():
        lat = sample["lat"].float().to(device)
        lon = sample["lon"].float().to(device)
        t = sample.get("t")
        if t is not None:
            t = t.float().to(device)
        pred = lit_model.model(lat, lon, t)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = pred.reshape(-1)
    return pred.cpu()


def _plot_pass(sample, pred, *, out_path: Path) -> None:
    lat = sample["lat"].float().cpu().numpy()
    lon = sample["lon"].float().cpu().numpy()
    truth = sample["y"].float().cpu().numpy()
    pred_np = pred.cpu().numpy()

    combined = np.concatenate([truth, pred_np])
    abs_vals = np.abs(combined)
    if abs_vals.size == 0:
        raise ValueError("No samples available to plot.")
    clip_low = np.percentile(abs_vals, 2)
    clip_high = np.percentile(abs_vals, 98)
    hi = max(clip_high, 1e-6)
    if np.any(combined < 0):
        vmin = -hi
        vmax = hi
    else:
        vmin = clip_low
        vmax = hi

    diff = pred_np - truth
    diff_abs = np.abs(diff)
    diff_hi = max(np.percentile(diff_abs, 98), 1e-6)
    diff_vmin, diff_vmax = -diff_hi, diff_hi

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    sc_truth = axes[0].scatter(lon, lat, c=truth, s=6, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth SSH")
    axes[0].set_xlabel("Longitude (deg)")
    axes[0].set_ylabel("Latitude (deg)")

    sc_pred = axes[1].scatter(lon, lat, c=pred_np, s=6, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Predicted SSH")
    axes[1].set_xlabel("Longitude (deg)")

    sc_diff = axes[2].scatter(lon, lat, c=diff, s=6, cmap="seismic", vmin=diff_vmin, vmax=diff_vmax)
    axes[2].set_title("Prediction - Truth")
    axes[2].set_xlabel("Longitude (deg)")

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.3)

    cbar_main = fig.colorbar(
        sc_pred,
        ax=axes[:2],
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
        label="SSH (m)",
    )
    cbar_main.ax.tick_params(labelsize=8)

    cbar_diff = fig.colorbar(
        sc_diff,
        ax=axes[2],
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
        label="ΔSSH (m)",
    )
    cbar_diff.ax.tick_params(labelsize=8)

    title_bits = []
    if "cycle" in sample:
        title_bits.append(f"cycle {int(sample['cycle'])}")
    if "pass" in sample:
        title_bits.append(f"pass {int(sample['pass'])}")
    if title_bits:
        fig.suptitle(f"SWOT {' '.join(title_bits)}", y=0.97)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.18, wspace=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot model predictions for a single test pass.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML (use config_used.yaml).")
    parser.add_argument(
        "--checkpoint",
        help="Path to the trained Lightning checkpoint. If omitted, best_model_path from final_metrics.txt is used.",
    )
    parser.add_argument(
        "--metrics",
        help="Optional metrics file to read best checkpoint from (defaults to config directory / final_metrics.txt).",
    )
    parser.add_argument("--pass-index", type=int, default=0, help="Index of the test pass to plot.")
    parser.add_argument(
        "--output",
        default="swot_test_pass.png",
        help="Where to save the plot (PNG).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference, e.g. cpu, mps, cuda:0",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
    else:
        metrics_path = Path(args.metrics).resolve() if args.metrics else cfg_path.parent / "final_metrics.txt"
        metrics_path = metrics_path.resolve()
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not provided and metrics file not found: {metrics_path}."
            )
        best_path: Path | None = None
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("best_model_path="):
                    raw = line.split("=", 1)[1].strip()
                    if raw:
                        best_path = Path(raw).expanduser()
                        break
        if best_path is None:
            raise RuntimeError(
                f"No best_model_path entry in metrics file: {metrics_path}."
            )
        ckpt_path = best_path.resolve()
        print(f"Using checkpoint from metrics file: {ckpt_path}")
    out_path = Path(args.output).resolve()

    cfg = _load_cfg(cfg_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(args.device)
    lit_model, datamodule = _load_model(cfg, ckpt_path, device)
    sample = _pick_pass(datamodule, args.pass_index)
    pred = _predict_pass(lit_model, sample)

    _plot_pass(sample, pred, out_path=out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
