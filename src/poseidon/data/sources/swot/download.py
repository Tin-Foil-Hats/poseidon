"""Functional utilities for retrieving SWOT granules via earthaccess."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, Tuple
from collections import defaultdict
import re

try:                                                                             
    import earthaccess                
except ImportError:                                              
    earthaccess = None

SN_HR = "SWOT_L2_HR_RASTER_2.0"
SN_HR_D = "SWOT_L2_HR_RASTER_D"
SN_LR = "SWOT_L2_LR_SSH_Expert_2.0"
SN_LR_D = "SWOT_L2_LR_SSH_EXPERT_D"

SHORTNAMES = {
    "hr": (SN_HR,),
    "lr": (SN_LR,),
    "lr_d": (SN_LR_D,),
    "lr_any": (SN_LR_D, SN_LR),
}

SOURCE_RANK = {SN_LR_D: 2, SN_LR: 1, SN_HR: 1, SN_HR_D: 2}
PATTERN = re.compile(
    r"_(\d{8}T\d{6})_(\d{8}T\d{6})_([A-Z])([A-Z])([A-Z])([0-9A-Z])_(\d{2})"
)
ENV_RANK = {"P": 2, "D": 1}
FID_RANK = {"G": 3, "I": 2, "O": 1}


@dataclass(frozen=True)
class SWOTQuery:
    """Normalized search parameters for SWOT granules."""

    product_key: str
    bbox: Tuple[float, float, float, float]
    start: str
    end: str
    passes: Tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.product_key not in SHORTNAMES:
            raise ValueError(f"Unknown product '{self.product_key}'")
        if len(self.bbox) != 4:
            raise ValueError("bbox must be (min_lon, min_lat, max_lon, max_lat)")
        object.__setattr__(self, "bbox", tuple(float(b) for b in self.bbox))
        if self.passes:
            object.__setattr__(self, "passes", tuple(str(p) for p in self.passes))


@dataclass(frozen=True)
class SWOTDownloadPlan:
    """Planned filesystem and network actions for a SWOT download run."""

    query: SWOTQuery
    outdir: Path
    purged: Tuple[Path, ...]
    found: int
    filtered: int
    best: int
    to_download: Tuple[MutableMapping[str, Any], ...]
    to_delete: Tuple[Path, ...]


@dataclass(frozen=True)
class SWOTDownloadResult:
    """Execution result for a SWOT download run."""

    plan: SWOTDownloadPlan
    downloaded: Tuple[Path, ...]
    deleted: Tuple[Path, ...]
    dry_run: bool

    @property
    def purged(self) -> Tuple[Path, ...]:
        return self.plan.purged


@dataclass(frozen=True)
class SWOTDownloadConfig:
    """Convenience wrapper for config-driven download execution."""

    product: str
    bbox: Tuple[float, float, float, float]
    start: str
    end: str
    outdir: Path
    passes: Tuple[str, ...] | None = None
    dry_run: bool = False
    purge: bool = False

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "SWOTDownloadConfig":
        bbox = _coerce_bbox(cfg)
        passes = _coerce_passes(cfg)
        outdir = cfg.get("outdir") or cfg.get("output") or cfg.get("directory")
        if outdir is None:
            raise ValueError("Config missing 'outdir'/'output'/'directory'")
        product = cfg.get("product") or cfg.get("product_key")
        if product is None:
            raise ValueError("Config missing 'product' or 'product_key'")
        start = cfg.get("start") or cfg.get("start_date") or cfg.get("tmin")
        end = cfg.get("end") or cfg.get("end_date") or cfg.get("tmax")
        if not start or not end:
            raise ValueError("Config missing start/end values")
        return cls(
            product=str(product),
            bbox=bbox,
            start=str(start),
            end=str(end),
            outdir=Path(outdir),
            passes=passes,
            dry_run=bool(cfg.get("dry_run", False)),
            purge=bool(cfg.get("purge", False)),
        )

    def to_query(self) -> SWOTQuery:
        return SWOTQuery(
            product_key=self.product,
            bbox=self.bbox,
            start=self.start,
            end=self.end,
            passes=self.passes,
        )

    def run(self, *, api: Any | None = None) -> SWOTDownloadResult:
        return run_swot_download(
            self.to_query(),
            outdir=self.outdir,
            purge=self.purge,
            dry_run=self.dry_run,
            api=api,
        )


def _require_api(api: Any) -> Any:
    if api is None:
        raise RuntimeError("earthaccess is required; pass an API stub when testing")
    for attr in ("login", "search_data", "download"):
        if not hasattr(api, attr):
            raise AttributeError(f"API object missing '{attr}'")
    return api


def ensure_login(api: Any | None = None) -> None:
    api = _require_api(api or earthaccess)
    api.login()


def _with_source(record: Any, short_name: str) -> MutableMapping[str, Any]:
    if isinstance(record, MutableMapping):
        record.setdefault("_src_sn", short_name)
        return record
    copy = dict(record)
    copy.setdefault("_src_sn", short_name)
    return copy


def search_swot_granules(query: SWOTQuery, *, api: Any | None = None) -> Tuple[MutableMapping[str, Any], ...]:
    api = _require_api(api or earthaccess)
    records: list[MutableMapping[str, Any]] = []
    temporal = (query.start, query.end)
    for short_name in SHORTNAMES[query.product_key]:
        for record in api.search_data(short_name=short_name, bounding_box=query.bbox, temporal=temporal):
            records.append(_with_source(record, short_name))
    return tuple(records)


def _native_id(record: Mapping[str, Any]) -> str:
    return str(record.get("meta", {}).get("native-id", ""))


def _match_name(name: str) -> tuple[str | None, re.Match[str] | None]:
    match = PATTERN.search(name)
    return (match.group(1), match) if match else (None, None)


def _minor_rank(ch: str) -> int:
    return int(ch.upper(), 36)


def _major_rank(ch: str) -> int:
    return ord(ch.upper()) - ord("A")


def _crid_score(match: re.Match[str]) -> tuple[int, int, int, int, int]:
    env = match.group(3)
    fid = match.group(4)
    major = match.group(5)
    minor = match.group(6)
    counter = int(match.group(7))
    return (
        ENV_RANK.get(env, 0),
        FID_RANK.get(fid, 0),
        _major_rank(major),
        _minor_rank(minor),
        counter,
    )


def filter_passes(records: Iterable[MutableMapping[str, Any]], query: SWOTQuery) -> Tuple[MutableMapping[str, Any], ...]:
    if query.passes is None or query.product_key not in {"lr", "lr_d", "lr_any"}:
        return tuple(records)
    needles = tuple(f"_{p}_" for p in query.passes)
    filtered: list[MutableMapping[str, Any]] = []
    for record in records:
        name = _native_id(record)
        if any(n in name for n in needles):
            filtered.append(record)
    return tuple(filtered)


def select_best_by_timestamp(records: Iterable[MutableMapping[str, Any]]) -> Tuple[MutableMapping[str, Any], ...]:
    buckets: dict[str, list[tuple[MutableMapping[str, Any], re.Match[str]]]] = defaultdict(list)
    for record in records:
        name = _native_id(record)
        ts, match = _match_name(name)
        if ts and match:
            buckets[ts].append((record, match))
    if not buckets:
        return tuple(records)

    winners: list[MutableMapping[str, Any]] = []
    for ts, items in buckets.items():
        items.sort(
            key=lambda pair: (
                _crid_score(pair[1]),
                SOURCE_RANK.get(pair[0].get("_src_sn", ""), 0),
                _native_id(pair[0]),
            ),
            reverse=True,
        )
        winners.append(items[0][0])
    return tuple(winners)


def _scan_local_granules(directory: Path) -> Tuple[Path, ...]:
    if not directory.exists():
        return tuple()
    paths: list[Path] = []
    for path in directory.iterdir():
        if path.is_file():
            ts, match = _match_name(path.name)
            if ts and match:
                paths.append(path)
    return tuple(paths)


def _build_download_actions(
    outdir: Path,
    remote_best: Sequence[MutableMapping[str, Any]],
    local_paths: Sequence[Path],
) -> tuple[list[MutableMapping[str, Any]], list[Path]]:
    local_by_ts: dict[str, list[tuple[Path, re.Match[str]]]] = {}
    for path in local_paths:
        ts, match = _match_name(path.name)
        if ts and match:
            local_by_ts.setdefault(ts, []).append((path, match))

    remote_by_ts: dict[str, tuple[MutableMapping[str, Any], re.Match[str]]] = {}
    for record in remote_best:
        name = _native_id(record)
        ts, match = _match_name(name)
        if ts and match:
            remote_by_ts[ts] = (record, match)

    to_download: list[MutableMapping[str, Any]] = []
    to_delete: list[Path] = []

    timestamps = set(local_by_ts.keys()) | set(remote_by_ts.keys())
    for ts in timestamps:
        local_entries = local_by_ts.get(ts, [])
        remote_entry = remote_by_ts.get(ts)
        if not remote_entry:
            continue
        remote_record, remote_match = remote_entry
        remote_name = _native_id(remote_record)
        if not remote_name:
            continue
        remote_path = outdir / remote_name

        best_local = None
        if local_entries:
            best_local = max(local_entries, key=lambda item: _crid_score(item[1]))

        if best_local is None:
            if not remote_path.exists():
                to_download.append(remote_record)
            continue

        local_path, local_match = best_local
        score_local = _crid_score(local_match)
        score_remote = _crid_score(remote_match)

        if score_remote > score_local:
            if not remote_path.exists():
                to_download.append(remote_record)
            for path, _ in local_entries:
                if path.name != remote_name:
                    to_delete.append(path)
    return to_download, to_delete


def apply_download_plan(
    plan: SWOTDownloadPlan,
    *,
    dry_run: bool = False,
    api: Any | None = None,
) -> SWOTDownloadResult:
    api = _require_api(api or earthaccess)
    downloaded_paths = tuple(
        plan.outdir / _native_id(record) for record in plan.to_download if _native_id(record)
    )
    deleted_paths = tuple(plan.to_delete)

    if dry_run:
        return SWOTDownloadResult(plan=plan, downloaded=downloaded_paths, deleted=deleted_paths, dry_run=True)

    for path in plan.to_delete:
        try:
            path.unlink()
        except FileNotFoundError:
            continue

    if plan.to_download:
        api.download(list(plan.to_download), str(plan.outdir))

    return SWOTDownloadResult(plan=plan, downloaded=downloaded_paths, deleted=deleted_paths, dry_run=False)


def run_swot_download(
    query: SWOTQuery,
    outdir: Path | str,
    *,
    purge: bool = False,
    dry_run: bool = False,
    api: Any | None = None,
) -> SWOTDownloadResult:
    api = _require_api(api or earthaccess)
    ensure_login(api)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    existing_paths = _scan_local_granules(outdir)
    purged: Tuple[Path, ...] = tuple()
    if purge and existing_paths:
        purged = existing_paths
        if not dry_run:
            for path in purged:
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
        existing_paths = tuple()

    found = search_swot_granules(query, api=api)
    filtered = filter_passes(found, query)
    best = select_best_by_timestamp(filtered)
    to_download, to_delete = _build_download_actions(outdir, best, existing_paths)

    plan = SWOTDownloadPlan(
        query=query,
        outdir=outdir,
        purged=purged,
        found=len(found),
        filtered=len(filtered),
        best=len(best),
        to_download=tuple(to_download),
        to_delete=tuple(to_delete),
    )
    return apply_download_plan(plan, dry_run=dry_run, api=api)


def _coerce_bbox(cfg: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    if "bbox" in cfg:
        bbox = cfg["bbox"]
    else:
        bbox = (
            cfg.get("min_lon"),
            cfg.get("min_lat"),
            cfg.get("max_lon"),
            cfg.get("max_lat"),
        )
    if isinstance(bbox, Sequence) and len(bbox) == 4:
        return tuple(float(b) for b in bbox)
    raise ValueError("Config must provide bbox or min_/max_ lon/lat")


def _coerce_passes(cfg: Mapping[str, Any]) -> Tuple[str, ...] | None:
    passes = cfg.get("passes")
    if passes is None:
        return None
    return tuple(str(p) for p in passes)


def main(cfg: Mapping[str, Any], *, api: Any | None = None) -> SWOTDownloadResult:
    product_key = cfg.get("product") or cfg.get("product_key")
    if product_key is None:
        raise ValueError("Config missing 'product' or 'product_key'")
    start = cfg.get("start") or cfg.get("start_date") or cfg.get("tmin")
    end = cfg.get("end") or cfg.get("end_date") or cfg.get("tmax")
    if not start or not end:
        raise ValueError("Config missing start/end values")
    bbox = _coerce_bbox(cfg)
    passes = _coerce_passes(cfg)

    query = SWOTQuery(product_key=product_key, bbox=bbox, start=str(start), end=str(end), passes=passes)

    outdir = cfg.get("outdir") or cfg.get("output") or cfg.get("directory")
    if outdir is None:
        raise ValueError("Config missing 'outdir'/'output'/'directory'")

    dry_run = bool(cfg.get("dry_run", False))
    purge = bool(cfg.get("purge", False))

    return run_swot_download(query, outdir=Path(outdir), purge=purge, dry_run=dry_run, api=api)


__all__ = [
    "SWOTQuery",
    "SWOTDownloadPlan",
    "SWOTDownloadResult",
    "SWOTDownloadConfig",
    "ensure_login",
    "search_swot_granules",
    "filter_passes",
    "select_best_by_timestamp",
    "run_swot_download",
    "apply_download_plan",
    "main",
]
