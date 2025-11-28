                  
import numpy as np, xarray as xr, re
from pathlib import Path


from poseidon.data.sources.swot.preprocess import query_landmask_values_at_swot, filter_filenames_seasonality, load_multiple_seasonality_files

PASS_RE = re.compile(r"_(\d{3})_(\d{3})_\d{8}T\d{6}")

def parse_cycle_pass(path: str):
    m = PASS_RE.search(str(path))
    if not m:
        raise ValueError(f"bad filename: {path}")
    return int(m.group(1)), int(m.group(2))

def _get_first(ds, names):
    for n in names:
        if n in ds:
            return ds[n]
    raise KeyError(f"none of {names} in dataset")

def to_seconds_like(val, ref):
    t = ref
    if t.ndim == 0:
        t = np.broadcast_to(t, val.shape)
    elif t.ndim == 1 and val.ndim == 2 and val.shape[0] == t.shape[0]:
        t = np.broadcast_to(t[:, None], val.shape)
    elif t.ndim == 1 and val.ndim == 2 and val.shape[1] == t.shape[0]:
        t = np.broadcast_to(t[None, :], val.shape)
    elif t.ndim != val.ndim:
        t2d, _ = xr.broadcast(xr.DataArray(ref), xr.DataArray(val))
        t = np.asarray(t2d)

    t = np.asarray(t)
    if np.issubdtype(t.dtype, np.datetime64):
        t = t.astype("datetime64[s]").astype("int64")
    return t.astype(np.float32)

def _load_measurement(ds, data_type: str):
    mode = data_type.upper()
    if mode not in {"SSH", "SSHA"}:
        raise ValueError(f"Unsupported data_type '{data_type}'. Expected 'SSH' or 'SSHA'.")

    base = _get_first(
        ds,
        ["ssh_karin", "ssh_karin2"] if mode == "SSH" else ["ssha_karin", "ssha_karin2"],
    )
    xover = _get_first(ds, ["height_cor_xover"])

    if mode == "SSH":
        geoid = _get_first(ds, ["geoid"])
        return xr.apply_ufunc(
            lambda a, b, c: a + b - c,
            base,
            xover,
            geoid,
            dask="allowed",
        )

    return xr.apply_ufunc(
        lambda a, b: a + b,
        base,
        xover,
        dask="allowed",
    )


def build_shards(
    netcdf_files,
    bbox,
    out_dir,
    watermask_dir,
    downsampling_factor=65,
    data_type: str = "SSH",
):
    """
    For each input file: pick SSH/SSHA measurement, bbox filter, landmask filter, then write shard.
    Keeps points where landmask==12 or 255.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for f in netcdf_files:
        cyc, pas = parse_cycle_pass(f)
        f = Path(f)
        ds = xr.open_dataset(f)

        lat = _get_first(ds, ["latitude", "lat"])
        lon = _get_first(ds, ["longitude", "lon"])
                             
        lonw = ((lon % 360) + 180) % 360 - 180

        data_var = _load_measurement(ds, data_type)

        tim = _get_first(ds, ["time"])
        t = to_seconds_like(data_var.values, np.asarray(tim.values))
        m = (
            np.isfinite(lat) & np.isfinite(lonw) & np.isfinite(data_var) &
            (lonw >= bbox["min_lon"]) & (lonw <= bbox["max_lon"]) &
            (lat  >= bbox["min_lat"]) & (lat  <= bbox["max_lat"])
        ).values

        if not np.any(m):
            ds.close()
            continue

        lat_f = lat.values[m].astype(np.float32)
        lon_f = lonw.values[m].astype(np.float32)
        y_f   = data_var.values[m].astype(np.float32)
        t_f   = t[m].astype(np.float32)


        bbox_valid = [float(lon_f.min()), float(lat_f.min()), float(lon_f.max()), float(lat_f.max())]
        matched = filter_filenames_seasonality(directory=watermask_dir, target_bbox=bbox_valid)
        combined_data, transform = load_multiple_seasonality_files(matched, target_bbox=bbox_valid)

        lm_vals = query_landmask_values_at_swot(
            lon_f, lat_f, combined_data, transform, downsampling_factor=downsampling_factor
        )
        keep = (lm_vals == 12) | (lm_vals == 255)
        if not np.any(keep):
            ds.close()
            continue

        lat_f = lat_f[keep]
        lon_f = lon_f[keep]
        y_f   = y_f[keep]
        t_f   = t_f[keep]

        ds.close()

        out_name = f"shard_c{cyc:03d}_p{pas:03d}_{f.stem}.npz"
        out_path = out_dir / out_name
        np.savez_compressed(
            out_path,
            lat=lat_f, lon=lon_f, t=t_f, y=y_f,
            cycle=np.full(lat_f.size, cyc, np.int16),
            pas=np.full(lat_f.size, pas, np.int16),
        )
        written.append(str(out_path))

    return written