"""Quick helper to plot train/val curves from Lightning CSV logs.

Usage:
    python scripts/plot_metrics.py --metrics experiments/swot_ssha_siren_fourier/lightning_logs/version_0/metrics.csv --out plot.png

This reads Lightning's metrics.csv (written by CSVLogger) and plots selected
metrics versus epoch. Defaults to train_loss/val_loss.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple


def _load_metrics(path: Path, keys: Tuple[str, ...]) -> dict[str, List[Tuple[float, float]]]:
    data: dict[str, list[tuple[float, float]]] = {k: [] for k in keys}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = float(row.get("epoch", 0))
            except Exception:
                continue
            for k in keys:
                if k in row and row[k] not in ("", "nan", "None"):
                    try:
                        v = float(row[k])
                    except Exception:
                        continue
                    data[k].append((epoch, v))
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Lightning CSV metrics.")
    ap.add_argument("--metrics", required=True, help="Path to Lightning metrics.csv")
    ap.add_argument("--out", default="metrics_plot.png", help="Output PNG file")
    ap.add_argument(
        "--keys",
        nargs="+",
        default=["train_loss", "val_loss"],
        help="Metric columns to plot (defaults to train_loss val_loss)",
    )
    args = ap.parse_args()

    metrics_path = Path(args.metrics).expanduser().resolve()
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")

    data = _load_metrics(metrics_path, tuple(args.keys))
    try:
        import matplotlib.pyplot as plt 
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it in a venv, e.g.,\n"
            "python -m venv .venv && source .venv/bin/activate && pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, series in data.items():
        if not series:
            continue
        series.sort(key=lambda p: p[0])
        xs, ys = zip(*series)
        ax.plot(xs, ys, label=key)

    ax.set_xlabel("epoch")
    ax.set_ylabel("metric value")
    ax.grid(True, alpha=0.3)
    if any(data.values()):
        ax.legend()
    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
