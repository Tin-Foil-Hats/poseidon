"""Batch evaluator that wraps eval_gulf_map over multiple experiment folders."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


def _safe_label(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "-", text.strip())
    return cleaned.strip("-") or "timestamp"


def _iter_experiment_dirs(root: Path) -> Iterable[Path]:
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        if child.name.lower() == "checkpoints":
            continue
        yield child


def evaluate_all_gulf_maps(
    experiments_root: Path | str,
    *,
    timestamp: str,
    coverage_dir: Path | str,
    lat_range: Sequence[float] = (18.0, 31.0),
    lon_range: Sequence[float] = (-98.0, -80.0),
    resolution: float = 0.1,
    chunk_size: int = 65536,
    device: str = "cpu",
    highlight_coverage: bool = True,
    extra_eval_args: Sequence[str] | None = None,
    python_executable: str | None = None,
) -> list[Path]:
    """Generate Gulf maps for every experiment directory under ``experiments_root``.

    Returns the list of generated image paths.
    """
    root = Path(experiments_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment root not found: {root}")

    coverage = Path(coverage_dir).expanduser().resolve()
    if not coverage.is_dir():
        raise FileNotFoundError(f"Coverage directory not found: {coverage}")

    lat_args = [str(float(lat_range[0])), str(float(lat_range[1]))]
    lon_args = [str(float(lon_range[0])), str(float(lon_range[1]))]
    safe_stamp = _safe_label(timestamp)
    extra = list(extra_eval_args or [])
    exe = python_executable or sys.executable

    outputs: list[Path] = []
    eval_script = Path(__file__).resolve().parent / "eval_gulf_map.py"

    for run_dir in _iter_experiment_dirs(root):
        cfg_path = run_dir / "config_used.yaml"
        if not cfg_path.is_file():
            print(f"Skipping {run_dir}: missing config_used.yaml")
            continue

        metrics_path = run_dir / "final_metrics.txt"
        output_dir = Path("outputs") / "jitter" / run_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"map_{safe_stamp}.png"

        cmd = [
            exe,
            str(eval_script),
            "--config",
            str(cfg_path),
            "--output",
            str(output_path),
            "--time",
            str(timestamp),
            "--coverage-dir",
            str(coverage),
            "--lat-range",
            *lat_args,
            "--lon-range",
            *lon_args,
            "--resolution",
            str(resolution),
            "--chunk-size",
            str(chunk_size),
            "--device",
            device,
        ]

        if metrics_path.is_file():
            cmd.extend(["--metrics", str(metrics_path)])

        if highlight_coverage:
            cmd.append("--show-coverage-highlight")

        cmd.extend(extra)

        print(f"Evaluating {run_dir.name} -> {output_path}")
        subprocess.run(cmd, check=True)
        outputs.append(output_path)

    return outputs


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch eval_gulf_map over experiment folders.")
    parser.add_argument(
        "--experiments-root",
        default="experiments/swot_ssha_siren_jitter",
        help="Folder containing experiment subdirectories.",
    )
    parser.add_argument(
        "--timestamp",
        required=True,
        help="Timestamp passed to eval_gulf_map (ISO8601 or epoch seconds).",
    )
    parser.add_argument(
        "--coverage-dir",
        default="data/shards/swot_ssha/whole_shards",
        help="Directory with SWOT whole shards for coverage overlays.",
    )
    parser.add_argument(
        "--lat-range",
        nargs=2,
        type=float,
        default=(18.0, 31.0),
        metavar=("LAT_MIN", "LAT_MAX"),
    )
    parser.add_argument(
        "--lon-range",
        nargs=2,
        type=float,
        default=(-98.0, -80.0),
        metavar=("LON_MIN", "LON_MAX"),
    )
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-highlight", action="store_true", help="Disable coverage highlight panel.")
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to eval_gulf_map after a '--'.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    extra = []
    if args.extra:
        extra = [arg for arg in args.extra if arg != "--"]
    evaluate_all_gulf_maps(
        args.experiments_root,
        timestamp=args.timestamp,
        coverage_dir=args.coverage_dir,
        lat_range=args.lat_range,
        lon_range=args.lon_range,
        resolution=args.resolution,
        chunk_size=args.chunk_size,
        device=args.device,
        highlight_coverage=not args.no_highlight,
        extra_eval_args=extra,
    )


if __name__ == "__main__":
    main()
