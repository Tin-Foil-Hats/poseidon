"""Preprocessing helpers for SWOT shard construction."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple
import re

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window
from scipy.spatial import cKDTree

_TILE_RE = re.compile(r"_(\d+)([EW])_(\d+)([NS])v")


def query_landmask_values_at_swot(
    swot_lons: np.ndarray,
    swot_lats: np.ndarray,
    combined_data: np.ndarray,
    transform: Affine,
    *,
    downsampling_factor: int = 30,
) -> np.ndarray:
    """Sample the nearest landmask value for each SWOT point using a KD-tree."""
    if combined_data is None or transform is None:
        raise ValueError("Landmask raster data and transform are required")

    if downsampling_factor <= 0:
        raise ValueError("downsampling_factor must be positive")

    sampled = combined_data[::downsampling_factor, ::downsampling_factor]
    rows, cols = np.indices(sampled.shape)
    lons, lats = transform * (cols * downsampling_factor, rows * downsampling_factor)

    landmask_coords = np.column_stack((lons.ravel(), lats.ravel()))
    landmask_values = sampled.ravel()
    valid = ~np.isnan(landmask_values)
    if not np.any(valid):
        raise ValueError("Landmask raster contains only NaNs in sampled region")

    tree = cKDTree(landmask_coords[valid])
    swot_points = np.column_stack((swot_lons, swot_lats))
    _, nearest_idx = tree.query(swot_points)
    return landmask_values[valid][nearest_idx]


def parse_filename_seasonality(filename: str | Path) -> Tuple[float, float, float, float] | None:
    """Extract a 10x10 degree bounding box encoded in the landmask filename."""
    match = _TILE_RE.search(Path(filename).name)
    if not match:
        return None
    lon = int(match.group(1)) * (-1 if match.group(2) == "W" else 1)
    lat = int(match.group(3)) * (1 if match.group(4) == "N" else -1)
    return (float(lon), float(lat - 10), float(lon + 10), float(lat))


def is_within_bounding_box(file_bbox: Sequence[float], target_bbox: Sequence[float]) -> bool:
    """Return True when the file bounding box overlaps the requested bounding box."""
    return not (
        file_bbox[2] < target_bbox[0]
        or file_bbox[0] > target_bbox[2]
        or file_bbox[3] < target_bbox[1]
        or file_bbox[1] > target_bbox[3]
    )


def filter_filenames_seasonality(directory: Path | str, target_bbox: Sequence[float]) -> list[str]:
    """Return filenames of watermask tiles overlapping the provided bounding box."""
    directory = Path(directory)
    matches: list[str] = []
    for path in sorted(directory.glob("*.tif")):
        file_bbox = parse_filename_seasonality(path)
        if file_bbox and is_within_bounding_box(file_bbox, target_bbox):
            matches.append(str(path))
    return matches


def calculate_window_seasonality(dataset: rasterio.io.DatasetReader, target_bbox: Sequence[float]) -> Window | None:
    """Compute the raster window overlapping the target bounding box."""
    left, top, right, bottom = dataset.bounds.left, dataset.bounds.top, dataset.bounds.right, dataset.bounds.bottom
    bbox_left, bbox_bottom, bbox_right, bbox_top = target_bbox

    if bbox_left > right or bbox_right < left or bbox_top < bottom or bbox_bottom > top:
        return None

    col_start, row_start = ~dataset.transform * (max(left, bbox_left), min(top, bbox_top))
    col_end, row_end = ~dataset.transform * (min(right, bbox_right), max(bottom, bbox_bottom))

    col_start, row_start = int(col_start), int(row_start)
    col_end, row_end = int(col_end), int(row_end)
    return Window(col_start, row_start, col_end - col_start, row_end - row_start)


def load_multiple_seasonality_files(filenames: Iterable[str], target_bbox: Sequence[float]) -> tuple[np.ndarray | None, Affine | None]:
    """Load and merge landmask rasters that intersect the target bounding box."""
    combined_data: np.ndarray | None = None
    transform: Affine | None = None

    for filename in filenames:
        with rasterio.open(filename) as dataset:
            window = calculate_window_seasonality(dataset, target_bbox)
            if window is None:
                continue

            data = dataset.read(1, window=window)
            window_transform = dataset.window_transform(window)

            if combined_data is None:
                res_x, res_y = dataset.res
                width = int(np.ceil((target_bbox[2] - target_bbox[0]) / res_x))
                height = int(np.ceil((target_bbox[3] - target_bbox[1]) / abs(res_y)))
                combined_data = np.full((height, width), np.nan, dtype=data.dtype)
                transform = Affine(res_x, 0.0, target_bbox[0], 0.0, -abs(res_y), target_bbox[3])

            col_offset = int((window_transform.c - transform.c) / transform.a)
            row_offset = int((window_transform.f - transform.f) / transform.e)
            combined_data[
                row_offset : row_offset + data.shape[0],
                col_offset : col_offset + data.shape[1],
            ] = data

    return combined_data, transform
