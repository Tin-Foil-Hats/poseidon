                      
"""Poseidon experiment runner with scalar/grid overrides."""
from __future__ import annotations

import argparse
import copy
import itertools
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import yaml

from poseidon.training import LitRegressor, SetEpochOnIterable, create_datamodule_from_config


def _slug(text: str) -> str:
    """Convert arbitrary text into a filesystem-safe slug."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or "run"


def _fmt_value(value: Any) -> str:
    """Format numbers for inclusion in run-directory tokens."""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return str(int(value)) if value.is_integer() else f"{value:g}"
    return _slug(str(value))


def _section(cfg: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """Return the named mapping from cfg as a plain dict, or empty dict."""
    value = cfg.get(key, {}) if isinstance(cfg, Mapping) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _get(cfg: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Traverse a dotted path (e.g. 'data.batch_size') and return the value."""
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _set(cfg: Dict[str, Any], path: str, value: Any) -> None:
    """Assign a value inside cfg using a dotted path, creating maps as needed."""
    cur = cfg
    keys = path.split(".")
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _deepcopy_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of a configuration mapping."""
    return copy.deepcopy(cfg)


def _apply_base_paths(cfg: Dict[str, Any], *, config_dir: Path, run_cwd: Path) -> None:
    """Resolve relative paths in cfg against either the config file or run cwd."""
    data_fields = (
        "data.shards_dir",
        "data.train_batchshard_dir",
    )
    run_fields = (
        "experiment.output_root",
        "experiment.checkpoints_dir",
        "run.ckpt_dir",
    )

    for field in data_fields:
        value = _get(cfg, field)
        if isinstance(value, str):
            resolved = _resolve_path(value, base=config_dir)
            _set(cfg, field, str(resolved))

    for field in run_fields:
        value = _get(cfg, field)
        if isinstance(value, str):
            resolved = _resolve_path(value, base=run_cwd)
            _set(cfg, field, str(resolved))


def _model_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the model block from cfg, falling back to the root mapping."""
    model = cfg.get("model")
    if isinstance(model, Mapping):
        return dict(model)
    return dict(cfg)


def _pe_type(cfg: Mapping[str, Any]) -> str:
    """Return the configured positional encoder type for naming/logging."""
    model = _model_cfg(cfg)
    pe = model.get("pe", {}) if isinstance(model.get("pe"), Mapping) else {}
    if "type" in pe and pe["type"] is not None:
        return str(pe["type"])
    if "target" in pe and pe["target"] is not None:
        return str(pe["target"])
    return "pe"


def _net_type(cfg: Mapping[str, Any]) -> str:
    """Return the configured network type string."""
    model = _model_cfg(cfg)
    net = model.get("net", {}) if isinstance(model.get("net"), Mapping) else {}
    return str(net.get("type", "net"))


def _name_suffix(cfg: Mapping[str, Any]) -> str:
    """Fetch optional experiment/run name suffix for run naming."""
    for section in ("experiment", "run"):
        suffix = _section(cfg, section).get("name_suffix")
        if suffix:
            return str(suffix)
    return ""


def _run_name(cfg: Mapping[str, Any]) -> str:
    """Compose the run directory name from config metadata."""
    base = _section(cfg, "experiment").get("name")
    if not base:
        base = _section(cfg, "run").get("name")
    if not base:
        base = time.strftime("run-%Y%m%d-%H%M%S")
    suffix = _name_suffix(cfg)
    if suffix:
        base = f"{base}{suffix}"
    return _slug(str(base))


def _monitor(cfg: Mapping[str, Any]) -> str:
    """Return the metric to monitor, defaulting to val_loss."""
    for section in ("experiment", "run", "trainer"):
        monitor = _section(cfg, section).get("monitor")
        if monitor:
            return str(monitor)
    return "val_loss"


def _mode(cfg: Mapping[str, Any]) -> str:
    """Return the optimization mode (min/max) for monitored metric."""
    for section in ("experiment", "run", "trainer"):
        mode = _section(cfg, section).get("mode")
        if mode:
            return str(mode)
    return "min"


def _early_stopping_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the early stopping configuration block if provided."""
    for section in ("experiment", "run", "trainer"):
        block = _section(cfg, section).get("early_stopping")
        if isinstance(block, Mapping):
            return dict(block)
    return {}


def _arch_subdir(cfg: Mapping[str, Any]) -> str:
    """Build a subdirectory name summarising architecture hyperparameters."""
    pe_type = _pe_type(cfg).lower()
    model = _model_cfg(cfg)
    pe = model.get("pe", {}) if isinstance(model.get("pe"), Mapping) else {}
    net = model.get("net", {}) if isinstance(model.get("net"), Mapping) else {}

    parts = []
    if pe_type == "fourier":
        n_lat = pe.get("n_lat")
        if n_lat is not None:
            parts.append(f"N{n_lat}")

    def _append_dims(prefix: str, block: Mapping[str, Any] | Any) -> bool:
        if not isinstance(block, Mapping):
            return False
        width = block.get("width")
        depth = block.get("depth")
        if width is not None:
            parts.append(f"{prefix}W{width}" if prefix else f"W{width}")
        if depth is not None:
            parts.append(f"{prefix}D{depth}" if prefix else f"D{depth}")
        return width is not None or depth is not None

    def _append_omega_tokens(prefix: str, block: Mapping[str, Any] | Any) -> None:
        if not isinstance(block, Mapping):
            return
        omega_initial = block.get("omega0_initial")
        if omega_initial is None and block.get("omega0") is not None:
            omega_initial = block.get("omega0")
        if omega_initial is not None:
            token = f"{prefix}Oi{_fmt_value(omega_initial)}" if prefix else f"Oi{_fmt_value(omega_initial)}"
            parts.append(token)
        omega_hidden = block.get("omega0_hidden")
        if omega_hidden is not None:
            token = f"{prefix}Oh{_fmt_value(omega_hidden)}" if prefix else f"Oh{_fmt_value(omega_hidden)}"
            parts.append(token)

    recorded = False
    recorded |= _append_dims("S", net.get("space_net"))
    recorded |= _append_dims("T", net.get("time_net"))
    if not recorded:
        _append_dims("", net)

    _append_omega_tokens("S", net.get("space_net"))
    _append_omega_tokens("T", net.get("time_net"))
    _append_omega_tokens("", net)

    parts.extend(_pe_tokens(pe_type, pe))
    parts.extend(_loss_weight_tokens(cfg))
    parts.extend(_data_loader_tokens(_section(cfg, "data")))
    parts.extend(_regularization_tokens(_section(cfg, "regularization")))

    return _slug("_".join(parts)) if parts else ""


def _pe_tokens(pe_type: str, pe_cfg: Mapping[str, Any]) -> list[str]:
    """Create naming tokens for positional encoder hyperparameters."""
    tokens: list[str] = []
    if not isinstance(pe_cfg, Mapping):
        return tokens

    if pe_type == "fourier_physical":
        spatial_scales = pe_cfg.get("spatial_scales_km")
        if isinstance(spatial_scales, Sequence) and not isinstance(spatial_scales, (str, bytes)):
            values = [float(s) for s in spatial_scales if isinstance(s, (int, float))]
            if values:
                tokens.append(f"FpS{len(values)}")
                tokens.append(f"FpSmin{_fmt_value(min(values))}")
                tokens.append(f"FpSmax{_fmt_value(max(values))}")

        time_scales = pe_cfg.get("time_scales_hours")
        if isinstance(time_scales, Sequence) and not isinstance(time_scales, (str, bytes)):
            values = [float(s) for s in time_scales if isinstance(s, (int, float))]
            if values:
                tokens.append(f"FpT{len(values)}")
                tokens.append(f"FpTmax{_fmt_value(max(values))}")

        time_norm = pe_cfg.get("time_norm")
        if time_norm:
            tokens.append(f"FpTn{_slug(str(time_norm))}")

        include_xyz = pe_cfg.get("include_xyz")
        if include_xyz is not None:
            tokens.append("FpXYZ" if bool(include_xyz) else "FpNoXYZ")

    elif pe_type == "fourier_physical_random":
        spatial_features = pe_cfg.get("spatial_features")
        temporal_features = pe_cfg.get("temporal_features")
        if spatial_features is not None:
            tokens.append(f"FpRS{_fmt_value(spatial_features)}")
        if temporal_features is not None:
            tokens.append(f"FpRT{_fmt_value(temporal_features)}")

    return tokens


def _loss_weight_tokens(cfg: Mapping[str, Any]) -> list[str]:
    """Extract loss-weight hyperparameters for directory naming."""
    training = _section(cfg, "training")
    weights = training.get("loss_weights") if isinstance(training.get("loss_weights"), Mapping) else None
    if not isinstance(weights, Mapping):
        return []

    tokens: list[str] = []

    dist_cfg = weights.get("distance_to_coast")
    if isinstance(dist_cfg, Mapping):
        method = str(dist_cfg.get("method", "")).lower()
        value = None
        if method == "exp":
            value = dist_cfg.get("scale_km")
        elif method == "linear":
            value = dist_cfg.get("max_km") or dist_cfg.get("scale_km")
        else:
            value = dist_cfg.get("scale_km") or dist_cfg.get("max_km")
        if value is not None:
            tokens.append(f"dist_km{_fmt_value(value)}")

    cross_cfg = weights.get("cross_track_distance")
    if isinstance(cross_cfg, Mapping):
        value = cross_cfg.get("sigma_km")
        if value is None:
            value = cross_cfg.get("scale_km") or cross_cfg.get("max_km")
        if value is not None:
            tokens.append(f"cross_km{_fmt_value(value)}")

    return tokens


def _data_loader_tokens(cfg: Mapping[str, Any]) -> list[str]:
    """Capture data-loader modifiers for directory naming."""
    tokens: list[str] = []
    if not isinstance(cfg, Mapping):
        return tokens
    loader = cfg.get("train_loader")
    if loader:
        tokens.append(f"DL{_slug(str(loader))}")
    micro = cfg.get("train_micro_bs")
    if isinstance(micro, (int, float)) and micro not in (0, None):
        tokens.append(f"DLm{_fmt_value(micro)}")
    if cfg.get("train_files_are_batches"):
        tokens.append("DLFileBatch")
    return tokens


def _regularization_tokens(cfg: Mapping[str, Any]) -> list[str]:
    """Capture regularization hyperparameters for directory naming."""
    tokens: list[str] = []
    if not isinstance(cfg, Mapping):
        return tokens
    laplace_weight = cfg.get("laplace_weight")
    if isinstance(laplace_weight, (int, float)) and laplace_weight > 0:
        tokens.append(f"LapW{_fmt_value(laplace_weight)}")
        laplace_len = cfg.get("laplace_length_km")
        if laplace_len is not None:
            tokens.append(f"LapL{_fmt_value(laplace_len)}")
        laplace_k = cfg.get("laplace_k")
        if laplace_k is not None:
            tokens.append(f"LapK{_fmt_value(laplace_k)}")
        laplace_max = cfg.get("laplace_max_points")
        if laplace_max is not None:
            tokens.append(f"LapN{_fmt_value(laplace_max)}")
    return tokens


def _resolve_path(path: os.PathLike[str] | str, *, base: Path | None = None) -> Path:
    """Resolve path relative to base (if given) or CWD, expanding user home."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    base_dir = base if base is not None else Path.cwd()
    return (base_dir / p).resolve()


def _run_dirs(cfg: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    """Return tuple of (root_dir, leaf_dir, ckpt_dir) for a run."""
    run_name = _run_name(cfg)
    arch_dir = _arch_subdir(cfg)

    exp = _section(cfg, "experiment")
    output_root = exp.get("output_root")
    if output_root:
        base_root = _resolve_path(output_root)
        if base_root.name != run_name:
            root_dir = base_root / run_name
        else:
            root_dir = base_root
    else:
        root_dir = _resolve_path(Path("runs") / run_name)

    leaf_dir = root_dir / arch_dir if arch_dir else root_dir

    ckpt_cfg = exp.get("checkpoints_dir") or _get(cfg, "run.ckpt_dir")
    if ckpt_cfg:
        base_ckpt = _resolve_path(ckpt_cfg)
        ckpt_dir = base_ckpt / arch_dir if arch_dir else base_ckpt
    else:
        ckpt_dir = leaf_dir / "checkpoints"

    return root_dir, leaf_dir, ckpt_dir


def _extract_metric(metrics: Mapping[str, Any], names: Iterable[str]) -> float | None:
    """Pick the first available metric from names and coerce to float."""
    for name in names:
        if name in metrics:
            value = metrics[name]
            try:
                item = value.item()
            except AttributeError:
                item = value
            try:
                return float(item)
            except (TypeError, ValueError):
                return None
    return None


def _prepare_trainer_kwargs(
    cfg: Mapping[str, Any],
    *,
    epochs: int | None,
    accelerator: str,
    devices: str,
    log_every: int,
    default_root: Path,
) -> Dict[str, Any]:
    """Merge trainer config with CLI overrides before trainer construction."""
    trainer_cfg = dict(_section(cfg, "trainer"))
    if epochs is not None:
        trainer_cfg["max_epochs"] = epochs
    trainer_cfg["accelerator"] = accelerator
    trainer_cfg["devices"] = int(devices) if isinstance(devices, str) and devices.isdigit() else devices
    trainer_cfg["log_every_n_steps"] = log_every
    trainer_cfg["default_root_dir"] = str(default_root)
    trainer_cfg.pop("logger", None)
    trainer_cfg.pop("callbacks", None)
    return trainer_cfg


def _train_once(
    cfg: Dict[str, Any],
    *,
    epochs: int | None,
    accelerator: str,
    devices: str,
    log_every: int,
) -> None:
    """Train a single experiment instance defined by cfg."""
    seed = int(_get(cfg, "data.seed", 0))
    pl.seed_everything(seed, workers=True)

    datamodule = create_datamodule_from_config(cfg)
    datamodule.setup("fit")
    model = LitRegressor(cfg, datamodule.stats)

    target_state = datamodule.stats.get("target") if isinstance(datamodule.stats, Mapping) else None
    if target_state is not None:
        data_cfg = cfg.setdefault("data", {})
        existing = data_cfg.get("target_normalizer", {})
        if isinstance(existing, str):
            existing = {"type": existing}
        elif isinstance(existing, Mapping):
            existing = dict(existing)
        else:
            existing = {}
        existing.setdefault("type", target_state.get("type", "none"))
        existing["state"] = target_state
        data_cfg["target_normalizer"] = existing

    root_dir, leaf_dir, ckpt_dir = _run_dirs(cfg)
    leaf_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = leaf_dir / "config_used.yaml"
    with cfg_path.open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    monitor = _monitor(cfg)
    mode = _mode(cfg)
    run_label = leaf_dir.relative_to(root_dir).as_posix() if leaf_dir != root_dir else root_dir.name

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{run_label}_ep{{epoch:03d}}",
        save_last=True,
        save_top_k=1,
        monitor=monitor,
        mode=mode,
        auto_insert_metric_name=False,
    )

    callbacks = [SetEpochOnIterable(), ckpt_cb]

    es_cfg = _early_stopping_cfg(cfg)
    if es_cfg.get("enabled"):
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=int(es_cfg.get("patience", 10)),
                min_delta=float(es_cfg.get("min_delta", 0.0)),
                verbose=bool(es_cfg.get("verbose", True)),
            )
        )

    logger = CSVLogger(save_dir=str(leaf_dir), name="logs")

    trainer_kwargs = _prepare_trainer_kwargs(
        cfg,
        epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        log_every=log_every,
        default_root=leaf_dir,
    )

    trainer = pl.Trainer(**trainer_kwargs, callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule=datamodule)

    metrics = trainer.callback_metrics
    train_last = _extract_metric(metrics, ("train_loss", "train_loss_epoch"))
    val_last = _extract_metric(metrics, ("val_loss", "val_loss_epoch"))
    best_score = None
    if ckpt_cb.best_model_score is not None:
        try:
            best_score = float(ckpt_cb.best_model_score.item())
        except AttributeError:
            best_score = float(ckpt_cb.best_model_score)

    report_path = leaf_dir / "final_metrics.txt"
    with report_path.open("w") as handle:
        handle.write(f"monitor={monitor}\n")
        handle.write(f"mode={mode}\n")
        handle.write(f"best_{monitor}={best_score}\n")
        handle.write(f"last_train_loss={train_last}\n")
        handle.write(f"last_val_loss={val_last}\n")
        handle.write(f"best_model_path={ckpt_cb.best_model_path}\n")


def _expand_grid(grid_specs: Sequence[str]) -> list[Dict[str, Any]]:
    """Expand --grid key=value1,value2 specs into individual override dicts."""
    axes: list[tuple[str, list[str]]] = []
    for spec in grid_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --grid spec: {spec}")
        key, values = spec.split("=", 1)
        choices = [val.strip() for val in values.split(",") if val.strip()]
        axes.append((key, choices))
    if not axes:
        return [dict()]
    keys = [key for key, _ in axes]
    value_lists = [vals for _, vals in axes]
    combos = []
    for combo in itertools.product(*value_lists):
        combos.append({k: v for k, v in zip(keys, combo)})
    return combos


def _coerce_scalar(value: str) -> Any:
    """Convert CLI override strings to bool/int/float when possible."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_overrides(pairs: Sequence[str]) -> Dict[str, Any]:
    """Parse --set key=value pairs into a flat override mapping."""
    overrides: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid --set value: {item}")
        key, value = item.split("=", 1)
        overrides[key] = _coerce_scalar(value)
    return overrides


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the Poseidon experiment runner CLI."""
    parser = argparse.ArgumentParser(description="Run Poseidon experiments over config grids.")
    parser.add_argument("--configs", nargs="+", help="YAML/JSON config files.")
    parser.add_argument("--set", nargs="*", default=[], help="Scalar overrides, e.g. data.batch_size=8192")
    parser.add_argument(
        "--grid",
        nargs="*",
        default=[],
        help="Comma-separated grid overrides, e.g. net.width=128,256",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max_epochs; leave unset to use config",
    )
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    args = parser.parse_args(argv)

    scalar_overrides = _parse_overrides(args.set)
    grid_overrides = _expand_grid(args.grid)

    for cfg_path in args.configs:
        cfg_file = Path(cfg_path).resolve()
        with cfg_file.open("r") as handle:
            base_cfg = yaml.safe_load(handle)
        if not isinstance(base_cfg, Mapping):
            raise ValueError(f"Configuration must be a mapping: {cfg_path}")

        for grid_vals in grid_overrides:
            cfg = _deepcopy_cfg(base_cfg)
            _apply_base_paths(cfg, config_dir=cfg_file.parent, run_cwd=Path.cwd())
            for key, value in scalar_overrides.items():
                _set(cfg, key, value)
            for key, value in grid_vals.items():
                _set(cfg, key, _coerce_scalar(value))
            _apply_base_paths(cfg, config_dir=cfg_file.parent, run_cwd=Path.cwd())

            _train_once(
                cfg,
                epochs=args.epochs,
                accelerator=args.accelerator,
                devices=args.devices,
                log_every=args.log_every_n_steps,
            )


if __name__ == "__main__":
    main()
