                      
"""Generate Gulf of Mexico SSH maps from a trained Poseidon model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

try:
	import imageio.v2 as imageio
except ImportError:
	try:
		import imageio
	except ImportError:
		imageio = None

try: 
	import cartopy.crs as ccrs 
	import cartopy.feature as cfeature 
except ImportError: 
	ccrs = None
	cfeature = None

from poseidon.data.schema import load_shard
from poseidon.data.shards import parse_cycle_pass_from_name
from poseidon.training import LitRegressor, create_datamodule_from_config


@dataclass(frozen=True)
class GridSpec:
	lat_values: np.ndarray
	lon_values: np.ndarray
	lat_points: np.ndarray
	lon_points: np.ndarray
	lat_indices: np.ndarray
	lon_indices: np.ndarray
	is_complete_mesh: bool
	point_times: np.ndarray | None = None
	extras: dict[str, np.ndarray] | None = None

	@property
	def point_count(self) -> int:
		return int(self.lat_points.size)


COLORBAR_PAD = 0.055
COLORBAR_FRACTION = 0.055
COVERAGE_SCATTER_SIZE = 4.0
COVERAGE_SCATTER_ALPHA = 0.4
MISFIT_CMAP = "seismic"
MISFIT_DEFAULT_LIMITS = (-0.2, 0.2)


def _get(cfg: Mapping[str, Any], path: str, default: Any = None) -> Any:
	cur: Any = cfg
	for part in path.split("."):
		if not isinstance(cur, Mapping) or part not in cur:
			return default
		cur = cur[part]
	return cur


def _set(cfg: dict[str, Any], path: str, value: Any) -> None:
	cur = cfg
	keys = path.split(".")
	for key in keys[:-1]:
		nxt = cur.get(key)
		if not isinstance(nxt, dict):
			nxt = {}
			cur[key] = nxt
		cur = nxt
	cur[keys[-1]] = value


def _resolve_path(path: str, *, base: Path) -> Path:
	candidate = Path(path).expanduser()
	if candidate.is_absolute():
		return candidate.resolve()
	return (base / candidate).resolve()


def _apply_base_paths(cfg: dict[str, Any], *, config_dir: Path) -> None:
	for field in ("data.shards_dir", "data.train_batchshard_dir"):
		value = _get(cfg, field)
		if isinstance(value, str):
			_set(cfg, field, str(_resolve_path(value, base=config_dir)))


def _load_grid_csv(csv_path: Path) -> GridSpec:
	csv_path = csv_path.expanduser().resolve()
	if not csv_path.exists():
		raise FileNotFoundError(f"Grid CSV not found: {csv_path}")
	data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8", ndmin=1)
	if data.size == 0:
		raise ValueError(f"Grid CSV {csv_path} has no rows")
	if data.dtype.names is None:
		raise ValueError(f"Grid CSV {csv_path} must include column headers")
	try:
		lat_arr = np.asarray(data["lat"], dtype=np.float64).reshape(-1)
		lon_arr = np.asarray(data["lon"], dtype=np.float64).reshape(-1)
	except (KeyError, ValueError) as exc:
		raise ValueError(f"Grid CSV {csv_path} must contain 'lat' and 'lon' columns") from exc
	lat_arr = lat_arr.astype(np.float32, copy=False)
	lon_arr = lon_arr.astype(np.float32, copy=False)
	if lat_arr.size == 0 or lon_arr.size == 0:
		raise ValueError(f"Grid CSV {csv_path} did not provide valid lat/lon values")
	lat_vals, lat_inv = np.unique(lat_arr, return_inverse=True)
	lon_vals, lon_inv = np.unique(lon_arr, return_inverse=True)
	coords = np.stack([lat_inv, lon_inv], axis=1)
	unique_pairs = np.unique(coords, axis=0)
	is_mesh = unique_pairs.shape[0] == lat_vals.size * lon_vals.size
	point_times: np.ndarray | None = None
	extras: dict[str, np.ndarray] = {}
	for name in data.dtype.names:
		if name in ("lat", "lon"):
			continue
		col = np.asarray(data[name])
		extras[name] = col.reshape(-1)
	for candidate in ("time", "t", "timestamp"):
		if candidate in extras:
			try:
				point_times = np.asarray(extras[candidate], dtype=np.float32).reshape(-1)
			except ValueError as exc:
				raise ValueError(f"Column '{candidate}' in {csv_path} must be numeric if provided") from exc
			break
	return GridSpec(
		lat_values=lat_vals.astype(np.float32, copy=False),
		lon_values=lon_vals.astype(np.float32, copy=False),
		lat_points=lat_arr.reshape(-1),
		lon_points=lon_arr.reshape(-1),
		lat_indices=lat_inv.astype(np.int64, copy=False),
		lon_indices=lon_inv.astype(np.int64, copy=False),
		is_complete_mesh=bool(is_mesh),
		point_times=point_times,
		extras=extras or None,
	)


def _make_rect_grid(lat_vals: np.ndarray, lon_vals: np.ndarray) -> GridSpec:
	lat_vals = np.asarray(lat_vals, dtype=np.float32)
	lon_vals = np.asarray(lon_vals, dtype=np.float32)
	if lat_vals.size == 0 or lon_vals.size == 0:
		raise ValueError("lat/lon axes must contain at least one entry")
	n_lat = int(lat_vals.size)
	n_lon = int(lon_vals.size)
	lat_points = np.repeat(lat_vals, n_lon)
	lon_points = np.tile(lon_vals, n_lat)
	lat_indices = np.repeat(np.arange(n_lat, dtype=np.int64), n_lon)
	lon_indices = np.tile(np.arange(n_lon, dtype=np.int64), n_lat)
	return GridSpec(
		lat_values=lat_vals,
		lon_values=lon_vals,
		lat_points=lat_points,
		lon_points=lon_points,
		lat_indices=lat_indices,
		lon_indices=lon_indices,
		is_complete_mesh=True,
	)


def _load_cfg(cfg_path: Path) -> dict[str, Any]:
	with cfg_path.open("r", encoding="utf-8") as handle:
		cfg = yaml.safe_load(handle)
	if not isinstance(cfg, Mapping):
		raise ValueError("Configuration must be a mapping")
	cfg_dict = dict(cfg)
	_apply_base_paths(cfg_dict, config_dir=cfg_path.parent)
	return cfg_dict


def _resolve_checkpoint(cfg_path: Path, checkpoint: str | None, metrics_path: str | None) -> Path:
	if checkpoint:
		return Path(checkpoint).expanduser().resolve()

	candidate = Path(metrics_path).expanduser() if metrics_path else cfg_path.parent / "final_metrics.txt"
	candidate = candidate.resolve()
	if candidate.exists():
		best_path: Path | None = None
		with candidate.open("r", encoding="utf-8") as handle:
			for line in handle:
				if line.startswith("best_model_path="):
					raw = line.split("=", 1)[1].strip()
					if raw:
						best_path = Path(raw).expanduser()
					break
		if best_path is not None:
			resolved = best_path.resolve()
			if resolved.exists():
				print(f"Using checkpoint from metrics file: {resolved}")
				return resolved
		print(f"best_model_path missing or invalid in metrics file: {candidate}")
	else:
		print(f"Metrics file not found: {candidate}")

	ckpt_dir = (cfg_path.parent / "checkpoints").resolve()
	if ckpt_dir.is_dir():
		ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
		if ckpts:
			print(f"Using most recent checkpoint from {ckpt_dir}: {ckpts[0]}")
			return ckpts[0].resolve()

	raise FileNotFoundError(
	 "Could not locate a checkpoint. Provide --checkpoint explicitly or ensure final_metrics.txt exists."
	)


def _load_model(cfg: dict[str, Any], ckpt_path: Path, device: torch.device):
	datamodule = create_datamodule_from_config(cfg)
	datamodule.setup("fit")
	datamodule.setup("test")
	lit_model = LitRegressor.load_from_checkpoint(
	 str(ckpt_path),
	 cfg=cfg,
	 dm_stats=datamodule.stats,
	 strict=False,
	)
	lit_model.eval()
	lit_model.to(device)
	return lit_model, datamodule


def _format_timestamp(ts: float | None) -> str:
	if ts is None:
		return "time: dataset mean"
	dt = datetime.fromtimestamp(ts, tz=timezone.utc)
	return dt.strftime("time: %Y-%m-%d %H:%M:%S UTC")


def _parse_time(raw: str | float | int | None, default: float | None) -> float | None:
	if raw is None:
		return default
	if isinstance(raw, (int, float)):
		return float(raw)
	text = str(raw).strip()
	if not text:
		return default
	try:
		return float(text)
	except ValueError:
		pass

	iso_text = text
	if iso_text.endswith("Z"):
		iso_text = iso_text[:-1] + "+00:00"
	dt = datetime.fromisoformat(iso_text)
	if dt.tzinfo is None:
		dt = dt.replace(tzinfo=timezone.utc)
	return dt.timestamp()


def _evaluate_grid(
 lit_model: LitRegressor,
 lat_vals: np.ndarray,
 lon_vals: np.ndarray,
 target_time: float | None,
 *,
 chunk_size: int,
) -> np.ndarray:
	device = next(lit_model.model.parameters()).device
	lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
	lat_flat = lat_grid.ravel().astype(np.float32, copy=False)
	lon_flat = lon_grid.ravel().astype(np.float32, copy=False)

	if target_time is not None:
		t_flat = np.full(lat_flat.shape, target_time, dtype=np.float32)
	else:
		t_flat = None

	preds: list[np.ndarray] = []
	total = lat_flat.size
	step = max(int(chunk_size), 1)
	for start in range(0, total, step):
		stop = min(start + step, total)
		lat_t = torch.from_numpy(lat_flat[start:stop]).to(device=device, dtype=torch.float32)
		lon_t = torch.from_numpy(lon_flat[start:stop]).to(device=device, dtype=torch.float32)
		if t_flat is not None:
			time_t = torch.from_numpy(t_flat[start:stop]).to(device=device, dtype=torch.float32)
		else:
			time_t = None
		with torch.no_grad():
			pred = lit_model.model(lat_t, lon_t, time_t)
			if isinstance(pred, tuple):
				pred = pred[0]
			pred = pred.reshape(-1)
		preds.append(pred.cpu().numpy())

	merged = np.concatenate(preds, axis=0)
	return merged.reshape(lat_grid.shape)


def _evaluate_points(
 lit_model: LitRegressor,
 lat: np.ndarray,
 lon: np.ndarray,
 time: np.ndarray | None,
 *,
 chunk_size: int,
) -> np.ndarray:
	device = next(lit_model.model.parameters()).device
	lat_flat = np.asarray(lat, dtype=np.float32).reshape(-1)
	lon_flat = np.asarray(lon, dtype=np.float32).reshape(-1)
	if time is not None:
		time_flat = np.asarray(time, dtype=np.float32).reshape(-1)
		if time_flat.shape != lat_flat.shape:
			raise ValueError("time array must match lat/lon shape")
	else:
		time_flat = None

	preds: list[np.ndarray] = []
	total = lat_flat.size
	step = max(int(chunk_size), 1)
	for start in range(0, total, step):
		stop = min(start + step, total)
		lat_t = torch.from_numpy(lat_flat[start:stop]).to(device=device, dtype=torch.float32)
		lon_t = torch.from_numpy(lon_flat[start:stop]).to(device=device, dtype=torch.float32)
		if time_flat is not None:
			time_t = torch.from_numpy(time_flat[start:stop]).to(device=device, dtype=torch.float32)
		else:
			time_t = None
		with torch.no_grad():
			pred = lit_model.model(lat_t, lon_t, time_t)
			if isinstance(pred, tuple):
				pred = pred[0]
			pred = pred.reshape(-1)
		preds.append(pred.cpu().numpy())

	return np.concatenate(preds, axis=0)


def _quantile_clamp(data: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
	flat = np.asarray(data, dtype=np.float64).reshape(-1)
	finite = flat[np.isfinite(flat)]
	if finite.size == 0:
		return -1.0, 1.0
	abs_vals = np.abs(finite)
	q_low = float(np.percentile(abs_vals, low))
	q_high = float(np.percentile(abs_vals, high))
	limit = max(q_high, 1e-6)
	if np.any(finite < 0):
		return -limit, limit
	return q_low, limit


def _symmetric_limits(data: np.ndarray, quantile: float = 98.0) -> tuple[float, float]:
	flat = np.asarray(data, dtype=np.float64).reshape(-1)
	finite = flat[np.isfinite(flat)]
	if finite.size == 0:
		return -1.0, 1.0
	abs_vals = np.abs(finite)
	hi = float(np.percentile(abs_vals, quantile))
	hi = max(hi, 1e-6)
	return -hi, hi


def _plot_frame(
 lat_vals: np.ndarray,
 lon_vals: np.ndarray,
 field: np.ndarray,
 out_path: Path,
 *,
 title: str,
 use_cartopy: bool,
 overlays: Iterable[dict[str, Any]] | None,
 color_limits: tuple[float, float] | None,
 show_misfit: bool,
 show_highlight: bool,
 point_data: dict[str, Any] | None,
 misfit_values: np.ndarray | None,
 misfit_limits: tuple[float, float] | None,
) -> None:
	overlays = list(overlays or [])
	point_lat = np.asarray(point_data.get("lat", [])) if point_data else np.array([])
	point_lon = np.asarray(point_data.get("lon", [])) if point_data else np.array([])
	point_vals = np.asarray(point_data.get("values", [])) if point_data else np.array([])
	has_points = point_lat.size > 0 and point_lon.size > 0
	if show_highlight and not has_points:
		show_highlight = False

	if show_misfit and (misfit_values is None or misfit_values.size == 0):
		show_misfit = False
	if show_misfit and not has_points:
		show_misfit = False

	if not show_misfit and not show_highlight:
		_plot_map(
		 lat_vals,
		 lon_vals,
		 field,
		 out_path,
		 title=title,
		 use_cartopy=use_cartopy,
		 overlays=overlays,
		 color_limits=color_limits,
		)
		return

	if color_limits is not None:
		vmin_field, vmax_field = color_limits
	else:
		vmin_field, vmax_field = _quantile_clamp(field)

	panel_count = 1 + int(show_highlight) + int(show_misfit)
	cmap_pred = "coolwarm"
	if use_cartopy and ccrs is not None and cfeature is not None:
		proj = ccrs.PlateCarree()
		fig = plt.figure(figsize=(8 * panel_count, 6))
		gs = fig.add_gridspec(1, panel_count, width_ratios=[1.0] * panel_count, wspace=0.12)
		axes = []
		for idx in range(panel_count):
			axis = fig.add_subplot(gs[0, idx], projection=proj)
			axis.coastlines(resolution="10m", color="black", linewidth=0.6)
			axis.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4)
			axis.add_feature(cfeature.LAND.with_scale("10m"), facecolor="0.9", alpha=0.5)
			axis.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="0.95", alpha=0.4)
			axis.set_extent(
			 [float(lon_vals.min()), float(lon_vals.max()), float(lat_vals.min()), float(lat_vals.max())],
			 crs=proj,
			)
			axes.append(axis)
	else:
		proj = None
		fig, axes = plt.subplots(1, panel_count, figsize=(8 * panel_count, 6), sharex=True, sharey=True)
		if panel_count == 1:
			axes = [axes]
		elif not isinstance(axes, (list, tuple)):
			axes = list(np.atleast_1d(axes))
		for axis in axes:
			axis.set_aspect("equal", adjustable="box")
			axis.set_xlabel("Longitude (deg)")
			axis.set_ylabel("Latitude (deg)")
			axis.grid(True, linewidth=0.4, alpha=0.35)

	ax_pred = axes[0]
	next_idx = 1
	ax_highlight = axes[next_idx] if show_highlight else None
	if show_highlight:
		next_idx += 1
	ax_misfit = axes[next_idx] if show_misfit else None

	if proj is not None:
		mesh_pred = ax_pred.pcolormesh(
			lon_vals,
			lat_vals,
			field,
			transform=proj,
			cmap=cmap_pred,
			shading="auto",
			vmin=vmin_field,
			vmax=vmax_field,
		)
	else:
		mesh_pred = ax_pred.pcolormesh(
			lon_vals,
			lat_vals,
			field,
			cmap=cmap_pred,
			shading="auto",
			vmin=vmin_field,
			vmax=vmax_field,
		)

	sc_highlight = None
	if ax_highlight is not None:
		if proj is not None:
			ax_highlight.pcolormesh(
				lon_vals,
				lat_vals,
				field,
				transform=proj,
				cmap="Greys",
				shading="auto",
				alpha=0.18,
			)
		else:
			ax_highlight.pcolormesh(
				lon_vals,
				lat_vals,
				field,
				cmap="Greys",
				shading="auto",
				alpha=0.18,
			)
		style = dict((point_data or {}).get("style", {})) if point_data else {}
		highlight_kwargs: dict[str, Any] = {
		 "s": float(style.get("size", 9.0)),
		 "alpha": float(style.get("alpha", 0.9)),
		 "linewidths": float(style.get("linewidths", 0.0)),
		}
		for key in ("marker", "edgecolors", "facecolors"):
			if key in style:
				highlight_kwargs[key] = style[key]
		if point_vals.size:
			highlight_kwargs.update(
			 {
			  "c": point_vals,
			  "cmap": cmap_pred,
			  "vmin": vmin_field,
			  "vmax": vmax_field,
			 }
			)
		else:
			highlight_kwargs["c"] = "tab:red"
		if proj is not None:
			highlight_kwargs["transform"] = proj
		sc_highlight = ax_highlight.scatter(point_lon, point_lat, **highlight_kwargs)
		if overlays:
			_apply_overlays(ax_highlight, overlays, proj, vmin_field, vmax_field, include_scatter=False)
		ax_highlight.set_title(f"{title} | Ground Truth")

	sc_misfit = None
	if ax_misfit is not None:
		if misfit_limits is None:
			misfit_limits = MISFIT_DEFAULT_LIMITS
		vmin_misfit, vmax_misfit = misfit_limits
		if proj is not None:
			ax_misfit.pcolormesh(
				lon_vals,
				lat_vals,
				field,
				transform=proj,
				cmap="Greys",
				shading="auto",
				alpha=0.3,
			)
		else:
			ax_misfit.pcolormesh(
				lon_vals,
				lat_vals,
				field,
				cmap="Greys",
				shading="auto",
				alpha=0.3,
			)
		base_style = dict((point_data or {}).get("style", {})) if point_data else {}
		misfit_size = float(base_style.get("size", 9.0))
		scatter_kwargs: dict[str, Any] = {
		 "c": misfit_values,
		 "cmap": MISFIT_CMAP,
		 "vmin": vmin_misfit,
		 "vmax": vmax_misfit,
		 "s": misfit_size,
		 "linewidths": float(base_style.get("linewidths", 0.0)),
		 "edgecolors": base_style.get("edgecolors", "none"),
		 "alpha": float(base_style.get("alpha", 0.85)),
		}
		for key in ("marker", "facecolors"):
			if key in base_style:
				scatter_kwargs[key] = base_style[key]
		if proj is not None:
			scatter_kwargs["transform"] = proj
		sc_misfit = ax_misfit.scatter(point_lon, point_lat, **scatter_kwargs)
		ax_misfit.set_title(f"{title} | Misfit (pred - obs)")

	if show_highlight and ax_highlight is not None and sc_highlight is not None:
		cbar_axes: Any = [ax_pred, ax_highlight]
	else:
		cbar_axes = ax_pred

	ax_pred.set_title(f"{title} | Prediction")
	cbar_pred = fig.colorbar(
		mesh_pred,
		ax=cbar_axes,
		orientation="horizontal",
		pad=COLORBAR_PAD,
		fraction=COLORBAR_FRACTION,
	)
	cbar_pred.set_label("SSHA (m)")
	if sc_misfit is not None and ax_misfit is not None:
		cbar_mis = fig.colorbar(
			sc_misfit,
			ax=ax_misfit,
			orientation="horizontal",
			pad=COLORBAR_PAD,
			fraction=COLORBAR_FRACTION,
		)
		cbar_mis.set_label("Pred - Obs (m)")

	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def _plot_map(
 lat_vals: np.ndarray,
 lon_vals: np.ndarray,
 field: np.ndarray,
 out_path: Path,
 *,
 title: str | None,
 use_cartopy: bool,
 overlays: Iterable[dict[str, Any]] | None,
 color_limits: tuple[float, float] | None,
) -> None:
	if color_limits is not None:
		vmin, vmax = color_limits
	else:
		vmin, vmax = _quantile_clamp(field)

	cmap_pred = "coolwarm"

	proj = None
	if use_cartopy and ccrs is not None and cfeature is not None:
		proj = ccrs.PlateCarree()
		fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": proj})
		mesh = ax.pcolormesh(
		 lon_vals,
		 lat_vals,
		 field,
		 transform=proj,
		 cmap=cmap_pred,
		 vmin=vmin,
		 vmax=vmax,
		)
		ax.coastlines(resolution="10m", color="black", linewidth=0.6)
		ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4)
		ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="0.9", alpha=0.5)
		ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="0.95", alpha=0.4)
		ax.set_extent([float(lon_vals.min()), float(lon_vals.max()), float(lat_vals.min()), float(lat_vals.max())], crs=proj)
	else:
		fig, ax = plt.subplots(figsize=(10, 6))
		mesh = ax.pcolormesh(
		 lon_vals,
		 lat_vals,
		 field,
		 cmap=cmap_pred,
		 shading="auto",
		 vmin=vmin,
		 vmax=vmax,
		)
		ax.set_aspect("equal", adjustable="box")
		ax.set_xlabel("Longitude (deg)")
		ax.set_ylabel("Latitude (deg)")
		ax.grid(True, linewidth=0.4, alpha=0.35)

	cbar = fig.colorbar(
		mesh,
		ax=ax,
		orientation="horizontal",
		pad=COLORBAR_PAD,
		fraction=COLORBAR_FRACTION,
	)
	cbar.set_label("SSHA (m)")

	if overlays:
		_apply_overlays(ax, overlays, proj, vmin, vmax)

	if title:
		ax.set_title(title)
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def _apply_overlays(
 ax: Any,
 overlays: Iterable[dict[str, Any]],
 proj: Any,
 vmin: float,
 vmax: float,
 *,
 include_scatter: bool = True,
) -> None:
	for item in overlays:
		kind = item.get("kind", "outline")
		if kind == "outline":
			lat_outline = item.get("lat")
			lon_outline = item.get("lon")
			if lat_outline is None or lon_outline is None:
				continue
			plot_kwargs: dict[str, Any] = {}
			if proj is not None:
				plot_kwargs["transform"] = proj
			ax.plot(
			 lon_outline,
			 lat_outline,
			 color=item.get("color", "white"),
			 alpha=float(item.get("alpha", 0.6)),
			 linewidth=float(item.get("linewidth", 1.2)),
			 **plot_kwargs,
			)
			label = item.get("label")
			if label:
				label_lat = float(np.mean(lat_outline))
				label_lon = float(np.mean(lon_outline))
				text_kwargs: dict[str, Any] = {
				 "ha": "center",
				 "va": "center",
				 "fontsize": 7,
				 "alpha": float(item.get("alpha", 0.6)),
				 "color": item.get("color", "white"),
				 "bbox": {"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.25},
				}
				if proj is not None:
					text_kwargs["transform"] = proj
				ax.text(label_lon, label_lat, str(label), **text_kwargs)
		elif include_scatter and kind == "scatter":
			lat_pts = item.get("lat")
			lon_pts = item.get("lon")
			if lat_pts is None or lon_pts is None:
				continue
			values = item.get("values")
			scatter_kwargs: dict[str, Any] = dict(item.get("scatter_kwargs", {}) or {})
			scatter_kwargs.update(
			 {
			  "c": values if values is not None else None,
			  "cmap": item.get("cmap", "coolwarm"),
			  "vmin": vmin,
			  "vmax": vmax,
			  "s": float(item.get("size", 6.0)),
			  "alpha": float(item.get("alpha", 0.6)),
			 }
			)
			if "linewidths" not in scatter_kwargs:
				scatter_kwargs["linewidths"] = 0.0
			if proj is not None and "transform" not in scatter_kwargs:
				scatter_kwargs["transform"] = proj
			ax.scatter(lon_pts, lat_pts, **scatter_kwargs)


def _scan_coverage_shards(shards_dir: Path) -> list[dict[str, Any]]:
	records: list[dict[str, Any]] = []
	for shard_path in sorted(shards_dir.rglob("*.npz")):
		try:
			cycle, ps = parse_cycle_pass_from_name(shard_path)
		except ValueError:
			cycle, ps = -1, -1
		try:
			arrays = load_shard(shard_path)
		except Exception:
			continue
		lat = np.asarray(arrays.get("lat"))
		lon = np.asarray(arrays.get("lon"))
		time = np.asarray(arrays.get("t"))
		if lat.size == 0 or lon.size == 0 or time.size == 0:
			continue
		lat_min = float(np.min(lat))
		lat_max = float(np.max(lat))
		lon_min = float(np.min(lon))
		lon_max = float(np.max(lon))
		t_min = float(np.min(time))
		t_max = float(np.max(time))
		outline_lat = np.array([lat_min, lat_min, lat_max, lat_max, lat_min], dtype=np.float32)
		outline_lon = np.array([lon_min, lon_max, lon_max, lon_min, lon_min], dtype=np.float32)
		records.append(
		 {
		  "path": shard_path,
		  "cycle": cycle,
		  "pass": ps,
		  "t_min": t_min,
		  "t_max": t_max,
		  "outline_lat": outline_lat,
		  "outline_lon": outline_lon,
		 }
		)
	return records


def _load_shard_points(
 shard_path: Path,
 cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	cached = cache.get(shard_path)
	if cached is not None:
		return cached
	with np.load(shard_path) as ds:
		lat = ds["lat"].astype(np.float32, copy=False)
		lon = ds["lon"].astype(np.float32, copy=False)
		y = ds["y"].astype(np.float32, copy=False)
		time = ds["t"].astype(np.float32, copy=False)
	cache[shard_path] = (lat, lon, y, time)
	return cache[shard_path]


def _build_overlays(
 records: Iterable[dict[str, Any]],
 target_time: float | None,
 *,
 window_seconds: float,
 color: str,
 alpha: float,
 linewidth: float,
 show_labels: bool,
 scatter_size: float = COVERAGE_SCATTER_SIZE,
 scatter_alpha: float = COVERAGE_SCATTER_ALPHA,
 cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
	records_list = list(records)
	if not records_list:
		return [], None

	half_window = window_seconds / 2.0 if window_seconds > 0 else None
	selected: list[dict[str, Any]] = []
	for rec in records_list:
		if target_time is None or half_window is None:
			include = True
		else:
			include = (rec["t_min"] <= target_time + half_window) and (rec["t_max"] >= target_time - half_window)
		if include:
			selected.append(rec)

	if not selected:
		return [], None

	overlays: list[dict[str, Any]] = []
	for rec in selected:
		label = None
		if show_labels and rec["cycle"] >= 0 and rec["pass"] >= 0:
			label = f"c{rec['cycle']:03d} p{rec['pass']:03d}"
		overlays.append(
		 {
		  "kind": "outline",
		  "lat": rec["outline_lat"],
		  "lon": rec["outline_lon"],
		  "color": color,
		  "alpha": alpha,
		  "linewidth": linewidth,
		  "label": label,
		 }
		)

	lat_blocks: list[np.ndarray] = []
	lon_blocks: list[np.ndarray] = []
	val_blocks: list[np.ndarray] = []
	time_blocks: list[np.ndarray] = []

	for rec in selected:
		lat, lon, val, time = _load_shard_points(rec["path"], cache)
		if target_time is None or half_window is None:
			sel_mask = None
		else:
			sel_mask = np.abs(time - target_time) <= half_window
		if sel_mask is None:
			sel_lat = lat
			sel_lon = lon
			sel_val = val
			sel_time = time
		else:
			if not np.any(sel_mask):
				continue
			sel_lat = lat[sel_mask]
			sel_lon = lon[sel_mask]
			sel_val = val[sel_mask]
			sel_time = time[sel_mask]
		if sel_lat.size == 0:
			continue
		lat_blocks.append(sel_lat)
		lon_blocks.append(sel_lon)
		val_blocks.append(sel_val)
		time_blocks.append(sel_time)

	point_data: dict[str, Any] | None = None
	if lat_blocks:
		lat_cat = np.concatenate(lat_blocks)
		lon_cat = np.concatenate(lon_blocks)
		val_cat = np.concatenate(val_blocks)
		time_cat = np.concatenate(time_blocks) if time_blocks else None
		overlays.append(
		 {
		  "kind": "scatter",
		  "lat": lat_cat,
		  "lon": lon_cat,
		  "values": val_cat,
		  "alpha": float(scatter_alpha),
		  "size": float(scatter_size),
		  "cmap": "coolwarm",
		  "scatter_kwargs": {
		   "edgecolors": "none",
		   "linewidths": 0.0,
		   "marker": ".",
		  },
		 }
		)
		point_data = {
		 "lat": lat_cat,
		 "lon": lon_cat,
		 "values": val_cat,
		 "time": time_cat if time_cat is not None else None,
		 "style": {
		  "size": float(scatter_size),
		  "alpha": float(scatter_alpha),
		  "marker": ".",
		  "linewidths": 0.0,
		  "edgecolors": "none",
		 },
		}

	return overlays, point_data


def _make_animation(frame_paths: list[Path], out_path: Path, fps: float) -> bool:
	if imageio is None:
		print("imageio is not installed; skipping animation.")
		return False
	if not frame_paths:
		return False
	try:
		fps_val = float(fps)
	except (TypeError, ValueError):
		fps_val = None
	if fps_val is None or not np.isfinite(fps_val) or fps_val <= 0:
		frame_delay_ms = 1000
	else:
		frame_delay_ms = max(int(round(1000.0 / fps_val)), 1)
	images = [imageio.imread(path) for path in frame_paths]
	per_frame_duration = [frame_delay_ms] * len(images)
	try:
		imageio.mimsave(out_path, images, duration=per_frame_duration, loop=0)
	except TypeError:
		imageio.mimsave(out_path, images, duration=frame_delay_ms, loop=0)
	return True


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate model over a Gulf of Mexico lat/lon grid.")
	parser.add_argument("--config", required=True, help="Path to config_used.yaml or base config.")
	parser.add_argument("--checkpoint", help="Optional checkpoint path (defaults to best_model_path).")
	parser.add_argument("--metrics", help="Optional metrics file location.")
	parser.add_argument("--output", default="gulf_map.png", help="Output PNG path.")
	parser.add_argument("--device", default="cpu", help="Device for inference (cpu, mps, cuda:0, ...).")
	parser.add_argument(
		"--grid-csv",
		help="Optional CSV with 'lat' and 'lon' columns defining evaluation points (overrides --lat-range/--lon-range/--resolution).",
	)
	parser.add_argument("--lat-range", nargs=2, type=float, default=[18.0, 31.0], help="Lat bounds (deg).")
	parser.add_argument("--lon-range", nargs=2, type=float, default=[-98.0, -80.0], help="Lon bounds (deg).")
	parser.add_argument("--resolution", type=float, default=0.1, help="Grid step in degrees.")
	parser.add_argument("--time", help="Scalar time (seconds) or ISO8601 (defaults to dataset mean).")
	parser.add_argument("--chunk-size", type=int, default=65536, help="Batch size for inference chunks.")
	parser.add_argument("--no-cartopy", action="store_true", help="Disable cartopy basemap even if available.")
	parser.add_argument("--time-count", type=int, default=1, help="Number of consecutive maps to generate.")
	parser.add_argument(
	 "--time-mode",
	 choices=["fixed", "coverage"],
	 default="fixed",
	 help="Select how timestamps are generated. 'coverage' iterates over shard coverage midpoints.",
	)
	parser.add_argument(
	 "--time-step-hours",
	 type=float,
	 default=24.0,
	 help="Hours between consecutive maps when --time-count > 1.",
	)
	parser.add_argument(
	 "--coverage-dir",
	 help="Optional shard directory to overlay coverage polygons (expects SWOT whole shard npz files).",
	)
	parser.add_argument(
	 "--max-frames",
	 type=int,
	 help="Optional cap on the number of frames when using time-mode=coverage.",
	)
	parser.add_argument(
	 "--coverage-window-hours",
	 type=float,
	 default=12.0,
	 help="Time window (hours) around each map timestamp when selecting shard coverage to overlay.",
	)
	parser.add_argument(
	 "--coverage-color",
	 default="white",
	 help="Color for shard coverage outlines.",
	)
	parser.add_argument(
	 "--coverage-alpha",
	 type=float,
	 default=0.6,
	 help="Alpha for shard coverage overlays.",
	)
	parser.add_argument(
	 "--coverage-linewidth",
	 type=float,
	 default=1.2,
	 help="Line width for shard coverage outlines.",
	)
	parser.add_argument(
	 "--coverage-labels",
	 action="store_true",
	 help="Annotate shard outlines with cycle/pass labels.",
	)
	parser.add_argument(
	 "--show-misfit",
	 action="store_true",
		help="Render an additional panel with prediction minus shard residuals.",
	)
	parser.add_argument(
		"--show-coverage-highlight",
		action="store_true",
		help="Render an additional panel that overlays ground-truth shard coverage versus the prediction.",
	)
	parser.add_argument(
	 "--misfit-limits",
	 nargs=2,
	 type=float,
	 metavar=("VMIN", "VMAX"),
	 help="Fixed color limits for the misfit scatter (pred - obs).",
	)
	parser.add_argument(
	 "--color-limits",
	 nargs=2,
	 type=float,
	 metavar=("VMIN", "VMAX"),
	 help="Fixed color limits to apply to every frame.",
	)
	parser.add_argument(
	 "--auto-color",
	 action="store_true",
	 help="Use per-frame quantile color limits instead of fixed limits.",
	)
	parser.add_argument(
	 "--animate",
	 help="Optional path for an animation (GIF) assembled from generated frames.",
	)
	parser.add_argument(
	 "--fps",
	 type=float,
	 default=1.0,
	 help="Frames per second for the animation when --animate is provided (lower slows playback).",
	)
	args = parser.parse_args()

	cfg_path = Path(args.config).resolve()
	ckpt_path = _resolve_checkpoint(cfg_path, args.checkpoint, args.metrics)
	if not ckpt_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

	cfg = _load_cfg(cfg_path)
	device = torch.device(args.device)
	lit_model, datamodule = _load_model(cfg, ckpt_path, device)

	if args.grid_csv:
		grid_spec = _load_grid_csv(Path(args.grid_csv))
	else:
		lat_min, lat_max = sorted(args.lat_range)
		lon_min, lon_max = sorted(args.lon_range)
		step = float(args.resolution)
		if step <= 0:
			raise ValueError("resolution must be positive")
		lat_vals = np.arange(lat_min, lat_max + step, step, dtype=np.float32)
		lon_vals = np.arange(lon_min, lon_max + step, step, dtype=np.float32)
		grid_spec = _make_rect_grid(lat_vals, lon_vals)

	lat_vals = grid_spec.lat_values
	lon_vals = grid_spec.lon_values
	lat_min = float(np.min(lat_vals))
	lat_max = float(np.max(lat_vals))
	lon_min = float(np.min(lon_vals))
	lon_max = float(np.max(lon_vals))

	stats_time = _get(datamodule.stats, "time") if hasattr(datamodule, "stats") else None
	time_default = float(stats_time.get("mean", 0.0)) if isinstance(stats_time, Mapping) else None

	if args.time_mode == "coverage":
		target_time = _parse_time(args.time, None)
	else:
		target_time = _parse_time(args.time, time_default)

	time_min = None
	time_max = None
	if isinstance(stats_time, Mapping):
		tmin = stats_time.get("min")
		tmax = stats_time.get("max")
		if isinstance(tmin, (int, float)) and isinstance(tmax, (int, float)):
			time_min = float(tmin)
			time_max = float(tmax)

	if args.auto_color:
		color_limits: tuple[float, float] | None = None
	elif args.color_limits is not None:
		color_limits = (float(args.color_limits[0]), float(args.color_limits[1]))
	else:
		color_limits = (-0.6, 0.6)

	base_out = Path(args.output).resolve()
	out_dir = base_out.parent
	out_dir.mkdir(parents=True, exist_ok=True)
	stem = base_out.stem
	suffix = base_out.suffix or ".png"

	use_cartopy = (not args.no_cartopy) and (ccrs is not None and cfeature is not None)

	coverage_records: list[dict[str, Any]] = []
	coverage_cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
	show_misfit = bool(args.show_misfit)
	show_highlight = bool(args.show_coverage_highlight)

	if args.coverage_dir:
		coverage_dir = Path(args.coverage_dir).expanduser().resolve()
		if coverage_dir.is_dir():
			coverage_records = _scan_coverage_shards(coverage_dir)
		else:
			print(f"Coverage directory not found: {coverage_dir}")

	if args.time_mode == "coverage":
		if not coverage_records:
			raise ValueError("time-mode=coverage requires a valid --coverage-dir with SWOT shards.")
		time_entries: list[dict[str, Any]] = []
		for rec in coverage_records:
			midpoint = 0.5 * (rec["t_min"] + rec["t_max"])
			if target_time is not None and midpoint < target_time:
				continue
			time_entries.append({"time": midpoint, "record": rec})
		if not time_entries:
			raise ValueError("No shard midpoints satisfied the --time filter when using time-mode=coverage.")
		time_entries.sort(
		 key=lambda item: (
		  float(item["time"]),
		  int(item["record"].get("cycle", -1)),
		  int(item["record"].get("pass", -1)),
		 )
		)
		if args.max_frames is not None and args.max_frames > 0:
			time_entries = time_entries[: int(args.max_frames)]
		time_values = [entry["time"] for entry in time_entries]
		time_contexts: list[dict[str, Any] | None] = time_entries
	else:
		if args.time_count <= 1:
			time_values = [target_time]
		else:
			if target_time is None:
				raise ValueError(
				 "time-count > 1 requires a valid base time (provide --time or ensure stats include mean)."
				)
			step_seconds = float(args.time_step_hours) * 3600.0
			if step_seconds <= 0:
				raise ValueError("time-step-hours must be positive when generating multiple maps.")
			time_values = [target_time + i * step_seconds for i in range(int(args.time_count))]
		time_contexts = [None] * len(time_values)
	window_seconds = max(float(args.coverage_window_hours), 0.0) * 3600.0
	if (show_misfit or show_highlight) and not coverage_records:
		print("Coverage-dependent view requested but no coverage directory was provided; falling back to prediction-only frames.")

	frame_paths: list[Path] = []
	for idx, ts in enumerate(time_values):
		context = time_contexts[idx] if idx < len(time_contexts) else None
		chunk = max(int(args.chunk_size), 1)
		if grid_spec.is_complete_mesh:
			field = _evaluate_grid(
				lit_model,
				lat_vals,
				lon_vals,
				ts,
				chunk_size=chunk,
			)
		else:
			if grid_spec.point_times is not None:
				# Respect per-point timestamps supplied in the CSV when available.
				point_time = grid_spec.point_times
			elif ts is not None:
				point_time = np.full(grid_spec.point_count, ts, dtype=np.float32)
			else:
				point_time = None
			pred_points = _evaluate_points(
				lit_model,
				grid_spec.lat_points,
				grid_spec.lon_points,
				point_time,
				chunk_size=chunk,
			)
			field = np.full((lat_vals.size, lon_vals.size), np.nan, dtype=np.float32)
			field[grid_spec.lat_indices, grid_spec.lon_indices] = pred_points.astype(np.float32, copy=False)

		overlays: list[dict[str, Any]] | None = None
		point_data: dict[str, Any] | None = None
		if coverage_records:
			overlays, point_data = _build_overlays(
			 coverage_records,
			 ts,
			 window_seconds=window_seconds,
			 color=args.coverage_color,
			 alpha=float(args.coverage_alpha),
			 linewidth=float(args.coverage_linewidth),
			 show_labels=bool(args.coverage_labels),
			 cache=coverage_cache,
			)

		misfit_values: np.ndarray | None = None
		misfit_limits: tuple[float, float] | None = None
		if show_misfit and point_data is not None:
			point_chunk = max(int(args.chunk_size // 4), 1)
			pred_points = _evaluate_points(
			 lit_model,
			 point_data["lat"],
			 point_data["lon"],
			 point_data.get("time"),
			 chunk_size=point_chunk,
			)
			misfit_values = pred_points - point_data["values"]
			if args.misfit_limits is not None:
				misfit_limits = (float(args.misfit_limits[0]), float(args.misfit_limits[1]))
			else:
				misfit_limits = None

		if len(time_values) == 1:
			frame_path = base_out
		else:
			if context and context.get("record"):
				rec = context["record"]
				shard_stem = rec["path"].stem if isinstance(rec.get("path"), Path) else str(rec.get("path", ""))
				safe_stem = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in shard_stem)
				frame_path = out_dir / f"{stem}_{idx:04d}_{safe_stem}{suffix}"
			else:
				frame_path = out_dir / f"{stem}_{idx:03d}{suffix}"

		title_bits = ["Poseidon SSHA", _format_timestamp(ts)]
		if context and context.get("record"):
			rec = context["record"]
			cycle = rec.get("cycle")
			ps = rec.get("pass")
			if isinstance(cycle, int) and cycle >= 0 and isinstance(ps, int) and ps >= 0:
				title_bits.append(f"c{cycle:03d} p{ps:03d}")
		if ts is not None and time_min is not None and time_max is not None:
			if ts < time_min or ts > time_max:
				title_bits.append("(extrapolated)")
		title = " | ".join(title_bits)

		_plot_frame(
		 lat_vals,
		 lon_vals,
		 field,
		 frame_path,
		 title=title,
		 use_cartopy=use_cartopy,
		 overlays=overlays,
		 color_limits=color_limits,
		 show_misfit=show_misfit,
		 show_highlight=show_highlight,
		 point_data=point_data,
		 misfit_values=misfit_values,
		 misfit_limits=misfit_limits,
		)

		frame_paths.append(frame_path)

	if args.animate:
		anim_path = Path(args.animate).expanduser().resolve()
		anim_path.parent.mkdir(parents=True, exist_ok=True)
		if _make_animation(frame_paths, anim_path, float(args.fps)):
			print(f"Saved animation to {anim_path}")


if __name__ == "__main__":
	main()
