"""Bayesian optimization driver for Poseidon experiments."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from typing import Any, Dict

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import optuna
import torch
import yaml

from poseidon.metrics import laplacian_penalty
from poseidon.training import LitRegressor, SetEpochOnIterable, create_datamodule_from_config
from scripts import run as run_script


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration file must contain a mapping")
    return cfg


def _sample_params(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    sampled: Dict[str, Any] = {}
    for key, spec in space.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Search space for '{key}' must be a mapping")
        kind = str(spec.get("type", "float")).lower()
        name = spec.get("name") or key.replace(".", "_")
        if kind in {"float", "logfloat", "loguniform"}:
            low = float(spec["low"])
            high = float(spec["high"])
            step = spec.get("step")
            sampled[key] = trial.suggest_float(name, low, high, step=step, log=(kind != "float" and kind != "uniform"))
        elif kind in {"int", "logint"}:
            low = int(spec["low"])
            high = int(spec["high"])
            step = int(spec.get("step", 1))
            sampled[key] = trial.suggest_int(name, low, high, step=step, log=(kind == "logint"))
        elif kind in {"categorical", "choice"}:
            choices = spec.get("choices")
            if not isinstance(choices, (list, tuple)) or not choices:
                raise ValueError(f"Categorical search space for '{key}' needs non-empty 'choices'")
            sampled[key] = trial.suggest_categorical(name, choices)
        else:
            raise ValueError(f"Unsupported search-space type '{kind}' for '{key}'")
    return sampled


def _ensure_suffix(cfg: Dict[str, Any], trial_tag: str) -> None:
    exp = cfg.setdefault("experiment", {})
    suffix = str(exp.get("name_suffix", ""))
    exp["name_suffix"] = f"{suffix}{trial_tag}" if suffix else trial_tag


def _write_trial_report(leaf_dir: Path, trial_info: Dict[str, Any]) -> None:
    report_path = leaf_dir / "bo_trial_summary.yaml"
    with report_path.open("w") as handle:
        yaml.safe_dump(trial_info, handle, sort_keys=False)


def _estimate_smoothness(
    model: LitRegressor,
    datamodule,
    *,
    batches: int,
    max_points: int,
    k: int,
    length_km: float,
) -> float:
    if batches <= 0:
        return 0.0
    datamodule.setup("validate")
    loader = datamodule.val_dataloader()
    if loader is None:
        return 0.0
    model.eval()
    device = next(model.parameters()).device
    scores: list[float] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            lat_cpu = batch["lat"].float()
            lon_cpu = batch["lon"].float()
            t_cpu = batch.get("t")
            if t_cpu is not None:
                t_cpu = t_cpu.float()
            lat = lat_cpu.to(device)
            lon = lon_cpu.to(device)
            t = t_cpu.to(device) if t_cpu is not None else None
            pred = model.model(lat, lon, t)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred_vals = pred.reshape(-1).detach().cpu()
            penalty = laplacian_penalty(
                lat_cpu,
                lon_cpu,
                pred_vals,
                max_points=max_points,
                k=k,
                length_km=length_km,
            )
            scores.append(float(penalty.detach().cpu()))
            if batch_idx + 1 >= batches:
                break
    return float(sum(scores) / len(scores)) if scores else 0.0


def _train_trial(
    cfg: Dict[str, Any],
    *,
    epochs: int | None,
    accelerator: str,
    devices: str,
    log_every: int,
    smooth_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    seed = int(run_script._get(cfg, "data.seed", 0))
    pl.seed_everything(seed, workers=True)

    datamodule = create_datamodule_from_config(cfg)
    datamodule.setup("fit")
    model = LitRegressor(cfg, datamodule.stats)

    root_dir, leaf_dir, ckpt_dir = run_script._run_dirs(cfg)
    leaf_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = leaf_dir / "config_used.yaml"
    with cfg_path.open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    monitor = run_script._monitor(cfg)
    mode = run_script._mode(cfg)
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
    es_cfg = run_script._early_stopping_cfg(cfg)
    if es_cfg.get("enabled") and epochs is not None and epochs <= 1:
        es_cfg = dict(es_cfg)
        es_cfg["enabled"] = False
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

    trainer_kwargs = run_script._prepare_trainer_kwargs(
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
    val_loss = run_script._extract_metric(metrics, ("val_loss", "val_loss_epoch"))
    best_score = None
    if ckpt_cb.best_model_score is not None:
        try:
            best_score = float(ckpt_cb.best_model_score.item())
        except AttributeError:
            best_score = float(ckpt_cb.best_model_score)
    best_metric = best_score if best_score is not None else (val_loss if val_loss is not None else float("inf"))

    best_path = ckpt_cb.best_model_path if ckpt_cb.best_model_path else ""
    if best_path:
        best_model = LitRegressor.load_from_checkpoint(
            best_path,
            cfg=cfg,
            dm_stats=datamodule.stats,
            map_location=torch.device("cpu"),
            strict=False,
        )
    else:
        best_model = copy.deepcopy(model).cpu()
    smoothness = _estimate_smoothness(
        best_model,
        datamodule,
        batches=int(smooth_cfg.get("batches", 0)),
        max_points=int(smooth_cfg.get("max_points", 512)),
        k=int(smooth_cfg.get("k", 8)),
        length_km=float(smooth_cfg.get("length_km", 50.0)),
    )

    report = {
        "val_metric": float(best_metric),
        "smoothness": float(smoothness),
        "leaf_dir": str(leaf_dir),
        "best_checkpoint": best_path,
    }
    return report


def _build_objective(
    base_cfg: Dict[str, Any],
    *,
    config_path: Path,
    bo_cfg: Dict[str, Any],
    args,
):
    config_dir = config_path.parent
    search_space = bo_cfg.get("search_space") or {}
    smooth_cfg = {
        "batches": int(bo_cfg.get("smoothness_batches", 0)),
        "max_points": int(bo_cfg.get("smoothness_max_points", bo_cfg.get("laplace_max_points", 512))),
        "k": int(bo_cfg.get("smoothness_k", bo_cfg.get("laplace_k", 8))),
        "length_km": float(bo_cfg.get("smoothness_length_km", bo_cfg.get("laplace_length_km", 50.0))),
    }
    smooth_weight_default = float(bo_cfg.get("smoothness_weight", 0.0))
    weight_range_cfg = bo_cfg.get("smoothness_weight_range")
    weight_range: tuple[float, float] | None
    if isinstance(weight_range_cfg, (list, tuple)) and len(weight_range_cfg) == 2:
        weight_range = (float(weight_range_cfg[0]), float(weight_range_cfg[1]))
    else:
        weight_range = None
    trial_prefix = str(bo_cfg.get("trial_suffix", "_bo"))
    max_epochs = args.max_epochs or bo_cfg.get("max_epochs")

    def objective(trial: optuna.Trial) -> float:
        cfg = run_script._deepcopy_cfg(base_cfg)
        run_script._apply_base_paths(cfg, config_dir=config_dir, run_cwd=Path.cwd())
        overrides = _sample_params(trial, search_space)
        for path, value in overrides.items():
            run_script._set(cfg, path, value)
        seed_base = int(bo_cfg.get("seed", 0))
        run_script._set(cfg, "data.seed", seed_base + trial.number)
        trial_tag = f"{trial_prefix}{trial.number:03d}"
        _ensure_suffix(cfg, trial_tag)
        cfg.pop("bayesopt", None)
        run_script._apply_base_paths(cfg, config_dir=config_dir, run_cwd=Path.cwd())

        result = _train_trial(
            cfg,
            epochs=max_epochs,
            accelerator=args.accelerator,
            devices=args.devices,
            log_every=args.log_every_n_steps,
            smooth_cfg=smooth_cfg,
        )

        alpha = smooth_weight_default
        if weight_range is not None:
            alpha = trial.suggest_float("smoothness_weight", weight_range[0], weight_range[1])

        objective_value = result["val_metric"] + alpha * result["smoothness"]
        trial.set_user_attr("val_metric", result["val_metric"])
        trial.set_user_attr("smoothness", result["smoothness"])
        trial.set_user_attr("leaf_dir", result["leaf_dir"])
        trial.set_user_attr("best_checkpoint", result["best_checkpoint"])
        trial.set_user_attr("overrides", overrides)
        trial.set_user_attr("alpha", alpha)

        leaf_dir = Path(result["leaf_dir"])
        trial_report = {
            "trial": trial.number,
            "objective": float(objective_value),
            "val_metric": result["val_metric"],
            "smoothness": result["smoothness"],
            "smoothness_weight": alpha,
            "overrides": overrides,
        }
        _write_trial_report(leaf_dir, trial_report)
        print(
            f"[Trial {trial.number}] val={result['val_metric']:.4e} smooth={result['smoothness']:.4e} "
            f"alpha={alpha:.3f} obj={objective_value:.4e}"
        )
        return objective_value

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bayesian optimization for a Poseidon config")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument("--study-name", default=None, help="Optuna study name")
    parser.add_argument("--storage", default=None, help="Optuna storage URL (e.g. sqlite:///study.db)")
    parser.add_argument("--n-trials", type=int, default=None, help="Override number of trials")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs per trial")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--log-every-n-steps", type=int, default=50)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_config(config_path)
    bo_cfg = cfg.get("bayesopt")
    if not isinstance(bo_cfg, dict):
        raise ValueError("Config is missing a 'bayesopt' block")

    base_cfg = copy.deepcopy(cfg)
    base_cfg.pop("bayesopt", None)

    n_trials = args.n_trials or int(bo_cfg.get("num_trials", 20))
    direction = "minimize" if str(bo_cfg.get("mode", "min")).lower() != "max" else "maximize"

    study_kwargs = {"direction": direction}
    if args.storage:
        study_kwargs["storage"] = args.storage
        study_kwargs["study_name"] = args.study_name or config_path.stem
        study_kwargs["load_if_exists"] = True
    objective = _build_objective(base_cfg, config_path=config_path, bo_cfg=bo_cfg, args=args)

    study = optuna.create_study(**study_kwargs)
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    summary = {
        "trial": best.number,
        "objective": best.value,
        "val_metric": best.user_attrs.get("val_metric"),
        "smoothness": best.user_attrs.get("smoothness"),
        "smoothness_weight": best.user_attrs.get("alpha"),
        "overrides": best.user_attrs.get("overrides"),
        "leaf_dir": best.user_attrs.get("leaf_dir"),
        "best_checkpoint": best.user_attrs.get("best_checkpoint"),
    }
    print("Best trial summary:\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
