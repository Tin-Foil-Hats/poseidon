                      
"""CLI wrapper for the functional SWOT downloader."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from poseidon.data.sources.swot.download import SHORTNAMES, SWOTDownloadConfig


def _load_config(path: Path) -> Mapping[str, Any]:
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml 
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML configs") from exc
        data = yaml.safe_load(text)
    else:
        import json

        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError("Config file must define a mapping of settings")
    return data


def _resolve_bbox(args, cfg: Mapping[str, Any]) -> Sequence[float]:
    if args.bbox is not None:
        return args.bbox
    if "bbox" in cfg:
        return cfg["bbox"]
    keys = ("min_lon", "min_lat", "max_lon", "max_lat")
    if all(k in cfg for k in keys):
        return [cfg[k] for k in keys]
    raise ValueError("Provide bbox via CLI or config (bbox/min_/max_ lon/lat)")


def _resolve_temporal(args, cfg: Mapping[str, Any]) -> tuple[str, str]:
    start = args.start or cfg.get("start") or cfg.get("start_date") or cfg.get("tmin")
    end = args.end or cfg.get("end") or cfg.get("end_date") or cfg.get("tmax")
    if not start or not end:
        raise ValueError("Missing temporal window (start/end)")
    return str(start), str(end)


def _resolve_passes(args, cfg: Mapping[str, Any]) -> tuple[str, ...] | None:
    passes: Sequence[Any] | None = args.passes if args.passes is not None else cfg.get("passes")
    if passes is None:
        return None
    return tuple(str(p) for p in passes)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SWOT granules using config overrides.")
    parser.add_argument("--product", choices=tuple(SHORTNAMES.keys()), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config file (JSON or YAML) providing bbox and date defaults",
    )
    parser.add_argument("--outdir", type=Path, required=True, help="Directory to write granules")
    parser.add_argument("--passes", type=int, nargs="*", default=None, help="Restrict LR products to pass numbers")
    parser.add_argument("--start", default=None, help="Override start (YYYY-MM-DD or ISO)")
    parser.add_argument("--end", default=None, help="Override end (YYYY-MM-DD or ISO)")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Override bounding box",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan actions without touching disk")
    parser.add_argument("--purge", action="store_true", help="Remove existing granules before planning")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = _load_config(args.config)

    bbox = _resolve_bbox(args, cfg)
    start, end = _resolve_temporal(args, cfg)
    passes = _resolve_passes(args, cfg)

    merged_cfg = dict(cfg)
    merged_cfg.update(
        {
            "product": args.product,
            "bbox": bbox,
            "start": start,
            "end": end,
            "passes": passes,
            "outdir": str(args.outdir),
            "dry_run": args.dry_run or bool(cfg.get("dry_run", False)),
            "purge": args.purge or bool(cfg.get("purge", False)),
        }
    )

    config = SWOTDownloadConfig.from_mapping(merged_cfg)
    result = config.run()

    print(
        "found={found} filtered={filtered} best={best} download={download} delete={delete}".format(
            found=result.plan.found,
            filtered=result.plan.filtered,
            best=result.plan.best,
            download=len(result.downloaded),
            delete=len(result.deleted),
        )
    )

    if result.purged:
        for path in result.purged:
            print(f"purged: {Path(path).name}")

    if result.deleted:
        for path in result.deleted:
            print(f"remove: {Path(path).name}")

    if result.downloaded:
        for path in result.downloaded:
            print(f"download: {Path(path).name}")

    if result.dry_run:
        print("dry run only; no filesystem changes were made")


if __name__ == "__main__":
    main()
