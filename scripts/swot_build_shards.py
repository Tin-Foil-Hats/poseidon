                      
"""CLI wrapper around the SWOT shard builder utilities."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from poseidon.data.sources.swot.shard_builder import build_shards


def _load_config(path: Path) -> Mapping[str, Any]:
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as exc: 
            raise RuntimeError("PyYAML is required to load YAML configs") from exc
        data = yaml.safe_load(text)
    else:
        import json

        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError("Config file must define a mapping of settings")
    return data


def _coerce_bbox(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    if "bbox" in cfg:
        seq = cfg["bbox"]
        if isinstance(seq, Mapping):
            keys = {"min_lon", "min_lat", "max_lon", "max_lat"}
            if not keys.issubset(seq):
                missing = sorted(keys - set(seq))
                raise ValueError(f"bbox mapping missing keys: {missing}")
            return {k: float(seq[k]) for k in keys}
        if isinstance(seq, Sequence) and len(seq) == 4:
            return {
                "min_lon": float(seq[0]),
                "min_lat": float(seq[1]),
                "max_lon": float(seq[2]),
                "max_lat": float(seq[3]),
            }
        raise ValueError("bbox must be a mapping with min_/max_ keys or a 4-element sequence")

    keys = ("min_lon", "min_lat", "max_lon", "max_lat")
    if all(k in cfg for k in keys):
        return {k: float(cfg[k]) for k in keys}
    raise ValueError("Config must provide bbox or explicit min_/max_ lon/lat fields")


def _collect_netcdf_files(directory: Path, glob_pattern: str) -> list[Path]:
    if "**" in glob_pattern:
        matches: Iterable[Path] = directory.glob(glob_pattern)
    else:
        matches = directory.glob(glob_pattern)
    files = sorted(p for p in matches if p.is_file())
    return files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SWOT shards from downloaded granules.")
    parser.add_argument("--config", type=Path, required=True, help="Config file (JSON or YAML)")
    parser.add_argument("--granules", type=Path, default=None, help="Override granule directory")
    parser.add_argument("--watermask", type=Path, default=None, help="Override watermask tile directory")
    parser.add_argument("--outdir", type=Path, default=None, help="Override shard output directory")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Override bounding box",
    )
    parser.add_argument("--glob", default=None, help="Glob pattern for granules (default: *.nc)")
    parser.add_argument("--downsampling", type=int, default=None, help="Landmask KD-tree downsampling factor")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many granules")
    parser.add_argument("--data-type", choices=("SSH", "SSHA"), default=None, help="Measurement to export")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = _load_config(args.config)

    bbox = (
        {
            "min_lon": float(args.bbox[0]),
            "min_lat": float(args.bbox[1]),
            "max_lon": float(args.bbox[2]),
            "max_lat": float(args.bbox[3]),
        }
        if args.bbox is not None
        else _coerce_bbox(cfg)
    )

    granule_root = args.granules or cfg.get("granule_dir") or cfg.get("outdir")
    if granule_root is None:
        raise ValueError("Provide granule directory via CLI or config (granule_dir/outdir)")
    granule_dir = Path(granule_root)
    if not granule_dir.exists():
        raise FileNotFoundError(f"Granule directory not found: {granule_dir}")

    watermask_root = args.watermask or cfg.get("watermask_dir")
    if watermask_root is None:
        raise ValueError("Provide watermask directory via CLI or config (watermask_dir)")
    watermask_dir = Path(watermask_root)
    if not watermask_dir.exists():
        raise FileNotFoundError(f"Watermask directory not found: {watermask_dir}")

    shard_outdir = Path(args.outdir) if args.outdir else Path(cfg.get("shard_outdir", "swot_shards"))
    downsampling = int(args.downsampling or cfg.get("downsampling_factor", 65))
    glob_pattern = args.glob or cfg.get("granule_glob", "*.nc")
    data_type = (args.data_type or cfg.get("data_type", "SSH")).upper()

    netcdf_files = _collect_netcdf_files(granule_dir, glob_pattern)
    if args.limit is not None:
        netcdf_files = netcdf_files[: args.limit]
    if not netcdf_files:
        raise FileNotFoundError(
            f"No granule files found in {granule_dir} matching pattern '{glob_pattern}'."
        )

    written = build_shards(
        [str(path) for path in netcdf_files],
        bbox=bbox,
        out_dir=shard_outdir,
        watermask_dir=watermask_dir,
        downsampling_factor=downsampling,
        data_type=data_type,
    )

    print(
        "granules={granules} written={written} outdir={outdir}".format(
            granules=len(netcdf_files),
            written=len(written),
            outdir=shard_outdir.resolve(),
        )
    )

    if written:
        for path in written:
            print(f"shard: {Path(path).name}")


if __name__ == "__main__":
    main()
