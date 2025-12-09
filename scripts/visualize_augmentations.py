"""Generate diagnostic plots illustrating training-time data augmentations."""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

AUG_DEFAULTS = {
    "lat_jitter_deg": 0.01,
    "lon_jitter_deg": 0.01,
    "time_jitter_seconds": 3600.0,
    "value_jitter_std": 0.02,
    "sample_dropout": 0.05,
}


def load_batch_sample(path: Path, batch_index: int = 0) -> dict[str, torch.Tensor]:
    """Load a single micro-batch from a pre-batched training shard."""
    data = torch.load(path, map_location="cpu")
    X = data["X"][batch_index].to(dtype=torch.float32)
    Y = data["Y"][batch_index].to(dtype=torch.float32).squeeze(-1)
    lat = X[:, 0]
    lon = X[:, 1]
    t = X[:, 2]
    return {"lat": lat, "lon": lon, "t": t, "y": Y}


def apply_augmentations(
    batch: dict[str, torch.Tensor],
    cfg: dict[str, float],
    *,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Replicate the Lightning datamodule augmentation pipeline."""
    rng = random.Random(seed)
    torch_rng = torch.Generator(device=batch["lat"].device)
    torch_rng.manual_seed(rng.randrange(2**31))

    lat = batch["lat"].clone()
    lon = batch["lon"].clone()
    t: Optional[torch.Tensor] = batch.get("t")
    if t is not None:
        t = t.clone()
    y = batch["y"].clone()

    deltas: dict[str, torch.Tensor] = {}

    lat_jitter = float(cfg.get("lat_jitter_deg", 0.0) or 0.0)
    if lat_jitter > 0.0:
        lat_noise = torch.randn(lat.shape, generator=torch_rng, device=lat.device, dtype=lat.dtype)
        lat = lat + lat_noise * lat_jitter
        deltas["lat"] = lat - batch["lat"]
    else:
        deltas["lat"] = torch.zeros_like(lat)

    lon_jitter = float(cfg.get("lon_jitter_deg", 0.0) or 0.0)
    if lon_jitter > 0.0:
        lon_noise = torch.randn(lon.shape, generator=torch_rng, device=lon.device, dtype=lon.dtype)
        lon = lon + lon_noise * lon_jitter
        deltas["lon"] = lon - batch["lon"]
    else:
        deltas["lon"] = torch.zeros_like(lon)

    time_jitter = float(cfg.get("time_jitter_seconds", 0.0) or 0.0)
    if time_jitter > 0.0 and t is not None:
        t_noise = torch.randn(t.shape, generator=torch_rng, device=t.device, dtype=t.dtype)
        t = t + t_noise * time_jitter
        deltas["t"] = t - batch["t"]
    else:
        deltas["t"] = torch.zeros_like(batch["t"]) if t is not None else torch.tensor([])

    value_jitter = float(cfg.get("value_jitter_std", 0.0) or 0.0)
    if value_jitter > 0.0:
        y_noise = torch.randn(y.shape, generator=torch_rng, device=y.device, dtype=y.dtype)
        y = y + y_noise * value_jitter
        deltas["y"] = y - batch["y"]
    else:
        deltas["y"] = torch.zeros_like(y)

    dropout = float(cfg.get("sample_dropout", 0.0) or 0.0)
    if dropout > 0.0:
        keep_mask = torch.rand(lat.shape[0], generator=torch_rng, device=lat.device) > dropout
        # Ensure at least one sample survives to avoid empty batches in the visualization.
        if not bool(keep_mask.any()):
            keep_mask[torch.randint(0, keep_mask.shape[0], (1,), generator=torch_rng)] = True
        lat = lat[keep_mask]
        lon = lon[keep_mask]
        y = y[keep_mask]
        deltas["lat"] = deltas["lat"][keep_mask]
        deltas["lon"] = deltas["lon"][keep_mask]
        deltas["y"] = deltas["y"][keep_mask]
        if t is not None:
            t = t[keep_mask]
            deltas["t"] = deltas["t"][keep_mask]
        deltas["keep_mask"] = keep_mask
    else:
        deltas["keep_mask"] = torch.ones(lat.shape[0], dtype=torch.bool)

    augmented = {"lat": lat, "lon": lon, "y": y}
    if t is not None:
        augmented["t"] = t
    return augmented, deltas


def make_plots(
    original: dict[str, torch.Tensor],
    augmented: dict[str, torch.Tensor],
    deltas: dict[str, torch.Tensor],
    output_path: Path,
    *,
    max_points: int = 4000,
) -> None:
    """Generate scatter + histogram visualizations describing the augmentation."""
    original_np = {k: v.detach().cpu().numpy() for k, v in original.items()}
    augmented_np = {k: v.detach().cpu().numpy() for k, v in augmented.items()}
    delta_np = {k: v.detach().cpu().numpy() for k, v in deltas.items() if k != "keep_mask"}

    kept = deltas.get("keep_mask")
    kept_fraction = float(kept.float().mean().item()) if kept is not None else 1.0

    # Align original values with the kept mask for fair comparison.
    if kept is not None:
        mask_np = kept.cpu().numpy().astype(bool)
        original_masked = {k: original[k][mask_np].detach().cpu().numpy() for k in original_np.keys()}
    else:
        original_masked = original_np

    def sample_indices(n: int) -> np.ndarray:
        return np.random.choice(n, size=min(n, max_points), replace=False)

    idx = sample_indices(original_masked["lat"].shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_scatter = axes[0, 0]
    ax_scatter.scatter(
        original_masked["lon"][idx],
        original_masked["lat"][idx],
        s=6,
        alpha=0.4,
        label="Original",
        color="#2E86AB",
    )
    ax_scatter.scatter(
        augmented_np["lon"][idx],
        augmented_np["lat"][idx],
        s=6,
        alpha=0.4,
        label="Augmented",
        color="#F18F01",
    )
    ax_scatter.set_xlabel("Longitude (deg)")
    ax_scatter.set_ylabel("Latitude (deg)")
    ax_scatter.set_title("Spatial Jitter (sampled subset)")
    ax_scatter.legend(loc="upper right")

    ax_lat = axes[0, 1]
    ax_lat.hist(delta_np["lat"], bins=50, color="#2E86AB", alpha=0.8)
    ax_lat.set_xlabel("Latitude Offset (deg)")
    ax_lat.set_ylabel("Count")
    ax_lat.set_title("Latitude Jitter Distribution")

    ax_time = axes[1, 0]
    if "t" in delta_np and delta_np["t"].size > 0:
        ax_time.hist(delta_np["t"] / 3600.0, bins=50, color="#6A4C93", alpha=0.8)
        ax_time.set_xlabel("Time Offset (hours)")
        ax_time.set_ylabel("Count")
        ax_time.set_title("Temporal Jitter Distribution")
    else:
        ax_time.axis("off")
        ax_time.text(0.5, 0.5, "No time jitter configured", ha="center", va="center")

    ax_val = axes[1, 1]
    ax_val.hist(delta_np["y"], bins=50, color="#F18F01", alpha=0.8)
    ax_val.set_xlabel("SSH Offset (m)")
    ax_val.set_ylabel("Count")
    ax_val.set_title("Target Value Jitter Distribution")

    fig.suptitle(
        "Training-Time Augmentations\n"
        f"Keep probability: {kept_fraction * 100:.1f}%  (drop {100 - kept_fraction * 100:.1f}%)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Poseidon data augmentations")
    parser.add_argument(
        "--batch-shard",
        type=Path,
        default=Path("data/shards/swot_ssha/gom_midpoint_random_batches/batchshard_00000.pt"),
        help="Path to a pre-batched training shard (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/augmentation_jitter.png"),
        help="Where to store the generated figure",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Which micro-batch to visualize inside the shard",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used for augmentation sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch = load_batch_sample(args.batch_shard, batch_index=args.batch_index)
    augmented, deltas = apply_augmentations(batch, AUG_DEFAULTS, seed=args.seed)
    make_plots(batch, augmented, deltas, args.output)
    print(f"Saved augmentation visualization to {args.output}")


if __name__ == "__main__":
    main()
