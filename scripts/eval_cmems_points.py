#!/usr/bin/env python3
"""Evaluate a Poseidon model on the fixed CMEMS coordinate/time grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
import torch

import eval_gulf_map  # Reuse existing loading/evaluation helpers.

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COORDS_CSV = REPO_ROOT / "cmems_gulf_coords.csv"
DEFAULT_TIMESTAMPS_CSV = REPO_ROOT / "cmems_timestamps.csv"


def _load_coords(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Coordinate CSV not found: {csv_path}")
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8", ndmin=1)
    if data.size == 0:
        raise ValueError(f"Coordinate CSV {csv_path} has no rows")
    if data.dtype.names is None or not {"lat", "lon"}.issubset(data.dtype.names):
        raise ValueError(f"Coordinate CSV {csv_path} must contain 'lat' and 'lon' headers")
    lat = np.asarray(data["lat"], dtype=np.float32).reshape(-1)
    lon = np.asarray(data["lon"], dtype=np.float32).reshape(-1)
    if lat.size != lon.size:
        raise ValueError("Latitude and longitude arrays must be the same length")
    return lat, lon


def _load_timestamps(csv_path: Path) -> list[tuple[str, float]]:
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Timestamp CSV not found: {csv_path}")
    timestamps: list[tuple[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Timestamp CSV {csv_path} must include a header row")
        field_candidates = [name for name in reader.fieldnames if name]
        if not field_candidates:
            raise ValueError(f"Timestamp CSV {csv_path} header is empty")
        # Prefer the canonical column name but fall back to the first available header.
        if "iso_utc" in reader.fieldnames:
            col_name = "iso_utc"
        else:
            col_name = field_candidates[0]
        for row in reader:
            iso_text = (row.get(col_name) or "").strip()
            if not iso_text:
                continue
            ts_val = eval_gulf_map._parse_time(iso_text, None)
            if ts_val is None:
                raise ValueError(f"Failed to parse timestamp '{iso_text}' from {csv_path}")
            timestamps.append((iso_text, float(ts_val)))
    if not timestamps:
        raise ValueError(f"Timestamp CSV {csv_path} did not yield any valid rows")
    return timestamps


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Poseidon model at CMEMS locations/times.")
    parser.add_argument("--config", required=True, help="Path to config_used.yaml or base config.")
    parser.add_argument("--checkpoint", help="Optional checkpoint path (defaults to best_model_path).")
    parser.add_argument("--metrics", help="Optional metrics file location.")
    parser.add_argument("--device", default="cpu", help="Device for inference (cpu, mps, cuda:0, ...).")
    parser.add_argument("--chunk-size", type=int, default=65536, help="Batch size for inference chunks.")
    parser.add_argument(
        "--coords-csv",
        default=str(DEFAULT_COORDS_CSV),
        help="Override CMEMS coordinate CSV (defaults to cmems_gulf_coords.csv).",
    )
    parser.add_argument(
        "--timestamps-csv",
        default=str(DEFAULT_TIMESTAMPS_CSV),
        help="Override CMEMS timestamp CSV (defaults to cmems_timestamps.csv).",
    )
    parser.add_argument(
        "--output",
        default="outputs/cmems/cmems_predictions.csv",
        help="Destination CSV for predictions.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    ckpt_path = eval_gulf_map._resolve_checkpoint(cfg_path, args.checkpoint, args.metrics)
    cfg = eval_gulf_map._load_cfg(cfg_path)
    device = torch.device(args.device)
    lit_model, _ = eval_gulf_map._load_model(cfg, ckpt_path, device)

    coord_path = Path(args.coords_csv)
    ts_path = Path(args.timestamps_csv)
    lat, lon = _load_coords(coord_path)
    timestamps = _load_timestamps(ts_path)

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunk = max(int(args.chunk_size), 1)
    total_steps = len(timestamps)
    print(f"Loaded {lat.size} coordinate points and {total_steps} timestamps.")
    print(f"Writing predictions to {out_path} ...")

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iso_utc", "unix_time", "lat", "lon", "prediction"])
        for idx, (iso_text, ts_val) in enumerate(timestamps, start=1):
            time_vec = np.full(lat.shape, ts_val, dtype=np.float32)
            preds = eval_gulf_map._evaluate_points(
                lit_model,
                lat,
                lon,
                time_vec,
                chunk_size=chunk,
            )
            for lat_val, lon_val, pred in zip(lat, lon, preds):
                writer.writerow([iso_text, float(ts_val), float(lat_val), float(lon_val), float(pred)])
            if idx == 1 or idx == total_steps or idx % 25 == 0:
                print(f"Processed {idx}/{total_steps} timestamps ({iso_text}).")

    print("Done.")


if __name__ == "__main__":
    main()
