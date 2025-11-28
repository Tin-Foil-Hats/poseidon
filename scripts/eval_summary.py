                      
"""Summarize top-performing runs from an experiment directory."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize top runs from evaluation_summary.csv")
    parser.add_argument("--experiment-dir", required=True, help="Experiment directory containing evaluation_summary.csv")
    parser.add_argument("--top", type=int, default=5, help="Number of top runs to display")
    return parser.parse_args()


def load_summary(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"evaluation_summary.csv not found at {path}")
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError("evaluation_summary.csv is empty")
    return rows


def sort_by_r2(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    def key(row: Dict[str, str]) -> float:
        try:
            return float(row.get("r2", "nan"))
        except ValueError:
            return float("nan")
    return sorted(rows, key=key, reverse=True)


def print_report(rows: List[Dict[str, str]], experiment_dir: Path, top: int) -> None:
    def _fmt(val: str) -> str:
        try:
            num = float(val)
        except (TypeError, ValueError):
            return val
        if not math.isfinite(num):
            return "nan"
        return f"{num:.4f}"

    print(f"Summary for {experiment_dir} (top {top} by R^2):")
    print("run\tr2\tcorr\tpix_corr_med\trmse\tmae\tbias")
    for row in rows[:top]:
        run = row.get("run", "?")
        r2 = _fmt(row.get("r2", "nan"))
        corr = _fmt(row.get("corr", "nan"))
        pix_corr_med = _fmt(row.get("per_pixel_corr_median", "nan"))
        rmse = _fmt(row.get("rmse", "nan"))
        mae = _fmt(row.get("mae", "nan"))
        bias = _fmt(row.get("bias", "nan"))
        print(f"{run}\t{r2}\t{corr}\t{pix_corr_med}\t{rmse}\t{mae}\t{bias}")


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.experiment_dir).expanduser().resolve()
    summary_path = exp_dir / "evaluation_summary.csv"
    rows = load_summary(summary_path)
    ranked = sort_by_r2(rows)
    print_report(ranked, exp_dir, args.top)


if __name__ == "__main__":
    main()
