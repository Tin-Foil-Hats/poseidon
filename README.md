# Poseidon Altimetry ML Toolkit

Poseidon provides end-to-end utilities for working with satellite altimetry data: downloading SWOT granules, filtering them against water masks, converting the observations into trainable shards, and training neural models that pair positional encoders with configurable regression networks. The codebase targets Python 3.9+ and centres on PyTorch/Lightning workflows.

## Quick Start

### Prerequisites
- Python 3.9 or newer
- Git, curl, and standard build tooling
- NASA Earthdata credentials. Either per-call input or stored in `~/.netrc` or exported via `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`.

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch lightning numpy xarray rasterio earthaccess pandas scipy pyproj matplotlib tqdm requests
pip install -e .
```

### Repository Layout
| Path | Purpose |
| --- | --- |
| `configs/` | Example download / experiment configs (non-example configs are ignored by Git). |
| `data/` | Workspace for downloaded granules, shards, and batch tensors (ignored by Git). |
| `notebooks/` | Interactive workflows; `rect_baseline_mlp_test.ipynb` walks through the full training stack. |
| `scripts/` | CLI entrypoints for downloads, sharding, experiment runs, evaluation, and plotting (`swot_download.py`, `swot_build_shards.py`, `reshard_random_train.py`, `run.py`, `eval_*.py`, etc.). |
| `src/poseidon/data/` | Shard schema, dataset helpers, and download/source adapters. |
| `src/poseidon/models/` | Positional encoders, regression nets, loss registry, and `poseidon_model.py`. |
| `src/poseidon/training/` | Lightning `ShardedDataModule`, `LitRegressor`, and callbacks. |
| `tests/` | Pytest suites (currently focused on data pipeline correctness). |

## Data Workflow

1. **Download SWOT granules**
   ```bash
   python scripts/swot_download.py \
       --product lr_any \
       --config configs/data/swot_example.yaml \
       --outdir data/example/granules
   ```
   The config supplies the bounding box and time window; CLI flags can override any field. Use `--dry-run` to inspect the plan without writing files.

2. **Build shards from NetCDF granules**
   ```bash
   python scripts/swot_build_shards.py \
       --config configs/data/swot_example.yaml \
       --granules data/example/granules \
       --watermask data/example/watermask \
       --outdir data/example/whole_shards
   ```
   This step loads each SWOT pass with xarray, filters against the Global Surface Water tiles, and writes validated `.npz` shards (lat/lon/t/y as float32) while recording cycle/pass metadata.

3. **Reshard for streaming training**
   ```bash
   python scripts/reshard_random_train.py \
       --src_dir data/example/whole_shards \
       --out_dir data/example/batch_shards \
       --seed 0 \
       --batch_size 16384 \
       --batches_per_file 64
   ```
   Output `.pt` files contain pre-batched tensors suitable for the iterable Lightning loader. The script also prints the cycle/pass splits applied to the original shards.

4. **Indexing & splits**
   - `poseidon.data.shards.load_index` auto-creates `index.json` for `.npz` shards and keeps metadata in sync.
   - `split_by_cycle_pass` groups shards by SWOT cycle/pass before sampling validation and test ratios.
   - `ShardedDataModule` combines these utilities; its `stats` field captures bbox and time statistics that the positional encoders consume.
  - To avoid recomputing the same splits on every launch, point `data.split_cache_path` (optionally with `data.split_cache_auto_write: true`) at a JSON file inside your experiment config. When the metadata (seed/ratios/index signature) matches, the datamodule loads the cached train/val/test lists instantly; otherwise it regenerates once and overwrites the cache for subsequent runs.

## Command-Line Interfaces

Poseidon exposes its major workflows through lightweight Python CLIs that mirror the steps above while keeping the underlying modules testable:

- `scripts/swot_download.py`: Download SWOT granules ranked by CRID/source. Accepts `--config` plus overrides like `--bbox` or `--tmin/tmax`. `--dry-run` lists selected passes without downloading.
- `scripts/swot_build_shards.py`: Convert NetCDF passes into validated shard `.npz` files; automatically fetches the Global Surface Water tiles when pointed at a watermask directory.
- `scripts/reshard_random_train.py`: Re-shuffle shard indices into pre-batched `.pt` tensors for iterable dataloaders. Supports train/val/test splits, deterministic seeds, and configurable micro-batch sizes.
- `scripts/run.py`: Single-run experiment driver. Reads one or more configs, applies `--set key=value` overrides, optional parameter grids, and honours config-defined `trainer.max_epochs` by default. Extra flags `--epochs`, `--accelerator`, `--devices`, and `--log_every_n_steps` provide runtime overrides when needed.
- `scripts/run_multi.py`: Batch multiple configs (or override combinations) sequentially; internally uses the same helpers exposed here.
- `scripts/eval_summary.py`, `scripts/eval_experiments.py`, `scripts/eval_gulf_map.py`, `scripts/plot_pass_cycles.py`, `scripts/plot_test_pass.py`: Post-training utilities for summarising metrics, generating evaluation CSVs, or plotting pass/target diagnostics.

All scripts support `python scripts/<name>.py --help` for their full option set. Paths in configs are resolved relative to the config file, while CLI overrides are resolved relative to the current working directory.

## SWOT Gulf recipes (SSH vs SSHA)

End-to-end commands we use for the Gulf of Mexico runs. Both configs target the same bbox/time window; they only differ in `data_type` and output directories.

- **Download granules (shared for SSH/SSHA)**  
  ```bash
  python scripts/swot_download.py \
      --config configs/data/swot_data.yaml \
      --outdir data/sources/swot
  ```

- **Build shards – SSH** (`configs/data/swot_data.yaml`, `data_type: SSH`, shards to `data/shards/swot_only/whole_shards`)  
  ```bash
  python scripts/swot_build_shards.py \
      --config configs/data/swot_data.yaml \
      --granules data/sources/swot \
      --watermask data/sources/watermask \
      --outdir data/shards/swot_only/whole_shards
  ```

- **Build shards – SSHA** (`configs/data/swot_ssha.yaml`, `data_type: SSHA`, shards to `data/shards/swot_only_ssha/whole_shards`)  
  ```bash
  python scripts/swot_build_shards.py \
      --config configs/data/swot_ssha.yaml \
      --granules data/sources/swot \
      --watermask data/sources/watermask \
      --outdir data/shards/swot_only_ssha/whole_shards
  ```

- **Reshard for streaming (SSH)** → pre-batched `.pt` for the iterable loader  
  ```bash
  python scripts/reshard_random_train.py \
      --src_dir data/shards/swot_only/whole_shards \
      --out_dir data/shards/swot_only/batch_shards \
      --seed 0 \
      --batch_size 16384 \
      --batches_per_file 64
  ```

- **Reshard for streaming (SSHA)**  
  ```bash
  python scripts/reshard_random_train.py \
      --src_dir data/shards/swot_only_ssha/whole_shards \
      --out_dir data/shards/swot_only_ssha/batch_shards \
      --seed 0 \
      --batch_size 16384 \
      --batches_per_file 64
  ```

These batch shards plug directly into `train_loader: stream` in the experiment YAMLs (e.g., `configs/experiments/swot_ssha_siren.yaml`), with `train_batchshard_dir` pointing at the corresponding `batch_shards` directory.

## Training Recipes

The canonical configuration lives in `configs/experiments/example_rect_mlp.yaml`. It demonstrates running pre-batched shards with a rectangular positional encoder feeding an MLP. Minimal training script:

```python
from pathlib import Path
import yaml
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from poseidon.training import create_datamodule_from_config, LitRegressor, SetEpochOnIterable

cfg = yaml.safe_load(Path("configs/experiments/example_rect_mlp.yaml").read_text())
dm = create_datamodule_from_config(cfg)
dm.setup("fit")
module = LitRegressor(cfg, dm.stats)

checkpoint_cb = ModelCheckpoint(
    dirpath=cfg["experiment"]["checkpoints_dir"],
    filename="rect-{epoch:02d}-{val_loss:.4f}", monitor="val_loss", mode="min"
)
trainer = pl.Trainer(
    **cfg.get("trainer", {}),
    callbacks=[SetEpochOnIterable(), checkpoint_cb, LearningRateMonitor("epoch")]
)
trainer.fit(module, datamodule=dm)
trainer.test(module, datamodule=dm, ckpt_path="best")
```

For exploratory runs, open `notebooks/rect_baseline_mlp_test.ipynb`. The notebook mirrors the steps above: loading the YAML config, inspecting loaders, tracing tensor shapes through the model, and running `Trainer.fit`/`Trainer.test` with CSV logging.

## Model Components

### Positional Encoders (`src/poseidon/models/pe`)
| Encoder | Key Features |
| --- | --- |
| `latlon_minmax` | Scales lat/lon (and optional time) to [-1, 1] using dataset min/max stats for a lightweight, fully deterministic coordinate feed. |
| `rect_baseline` | Normalises lat/lon within bbox, z-score time, and appends annual + configurable tidal harmonics. |
| `cossin_simple` | Sin/cos cycles of scaled lat/lon with optional raw coordinates and daily/annual phases. |
| `fourier` | Deterministic Fourier bases over lat/lon (and optionally time) plus optional XYZ embedding. |
| `rawxyz` | Converts geodetic lat/lon to geocentric XYZ with optional fixed-frequency time terms. |
| `rff` | Random Fourier features with learnable bandwidth; time channel normalised via dataset stats. |
| `rff_ard_bands` | Multi-band RFF with automatic relevance determination per input dimension. |
| `rff_tides` | RFF backbone augmented with explicit tidal harmonics (M2/S2/N2/K1/O1/M4, optional annual terms).

Each encoder implements `feat_dim()` and can `bind_context(ctx)` to absorb bbox/time statistics supplied by `ShardedDataModule`.

### Regression Nets (`src/poseidon/models/nets`)
| Net | Highlights |
| --- | --- |
| `mlp` | Dense MLP with selectable depth, width, activation (`relu`/`gelu`/`silu`), dropout, and optional log-variance head for heteroscedastic losses. |
| `resmlp` | LayerNorm + residual MLP blocks for deeper stability; same optional uncertainty head. |
| `siren` | Sinusoidal representation network with configurable `omega0`, optional learnable hidden frequencies, and uncertainty head support. |
| `siren_stable` | SiLU-based alternative to SIREN with Xavier init for improved stability.

Networks subclass `NetBase` and are registered via `@register_net`, enabling `poseidon.models.poseidon_model.build_model` to instantiate them from config.

### Losses (`src/poseidon/models/losses`)
| Loss | Behaviour |
| --- | --- |
| `mse` | Mean-squared error supporting optional `(prediction, log_variance)` tuples for heteroscedastic training. |
| `mae` | Mean-absolute error with the same uncertainty-aware formulation. |
| `huber` | Piecewise quadratic/linear loss with configurable `delta`; integrates with uncertainty heads when present.

The registry auto-discovers modules, so adding a new loss only requires decorating it with `@register_loss`.

## Optimizer Usage

All experiment configs share the same `optim` block; by default the Lightning module wraps the model with AdamW and optional EMA tracking. Typical YAML:

```yaml
optim:
  lr: 3e-4           # AdamW learning rate
  wd: 5e-5           # Weight decay applied to all parameters
  betas: [0.9, 0.999]
  ema: false         # Set true to keep an exponential moving average of weights
  ema_decay: 0.999   # Only read when ema=true
  schedule: warmcosine
  warmup_epochs: 40
  warmup_start_lr: 3e-5
  eta_min: 5e-5
```

- **Base optimizer**: `torch.optim.AdamW` with config-supplied `lr`, `wd`, and `betas`. These values can be overridden via CLI (`--set optim.lr=1e-4`).
- **Schedulers**: Selectable via `optim.schedule`. Options include:
  - `none`/`constant`: raw AdamW without LR decay.
  - `cosine`: vanilla cosine annealing across `trainer.max_epochs`.
  - `cosine_warmup`/`warmup_cosine`: Lightning `SequentialLR` that linearly warms from `warmup_start` (or `warmup_start_lr`) before cosine decay.
  - `warmcosine` (most configs): custom `LinearWarmupCosineAnnealingLR` that respects `warmup_epochs`, `warmup_start_lr`, `max_epochs`, and `eta_min` (floor LR).
  - `plateau`: `ReduceLROnPlateau` keyed on `val_loss` with tunable `factor`, `patience`, and `min_lr`.
  - `onecycle`: `OneCycleLR` over estimated stepping batches with `pct_start`, `div_factor`, and `final_div_factor`.
  - `step`/`multistep`: Multi-step decay with user-defined `milestones` and `gamma`.
- **Warmup semantics**: For `cosine_warmup` or `warmcosine`, make sure `warmup_epochs` is no larger than the total epoch budget; the helper clamps invalid values automatically but logging stays cleaner when they match intent.
- **EMA weights**: Flip `optim.ema: true` to keep a detached moving average of parameters (decay via `ema_decay`); helpful for stabilizing SIREN tails. EMA checkpoints are not saved separately, so run-side eval should call `trainer.validate()` before toggling EMA off.

Because schedulers live inside the Lightning module, every entry in the `optim` block can also be switched mid-experiment via `scripts/run.py --set`. When sweeping hyperparameters, include them in `--grid` (e.g., `--grid optim.omega0_initial=4,6,10`) and the runner will emit one sub-experiment per value.

## Downloader CLIs and Processing Pipeline

Fetch raw data and process to train-ready shards:

- SWOT granules  
  ```bash
  python scripts/swot_download.py \
      --product lr_any \
      --config configs/data/swot_data.yaml \
      --outdir data/sources/swot
  ```
  Overrides: `--bbox`, `--start/--end`, `--passes`, `--dry-run`, `--purge`.

- Sentinel-6 (LR/HR, strict replace by baseline)  
  ```bash
  python -m poseidon.data.sources.sentinel6.download \
      --product lr \
      --config configs/data/gulf_config.json \
      --outdir data/sources/sentinel6 \
      --dry-run
  ```
  Use `--product hr` for HR; optional `--passes`, `--bbox`, `--start/--end`. Drop `--dry-run` to download.

- Global Surface Water mask tiles (for SWOT land masking)  
  ```bash
  python -m poseidon.data.sources.watermask.download \
      data/sources/watermask \
      DSMW
  ```
  Second arg is the dataset name (e.g., `DSMW`).

- Build shards from SWOT NetCDF (SSH)  
  ```bash
  python scripts/swot_build_shards.py \
      --config configs/data/swot_data.yaml \
      --granules data/sources/swot \
      --watermask data/sources/watermask \
      --outdir data/shards/swot_only/whole_shards
  ```
  For SSHA: swap `--config configs/data/swot_ssha.yaml` and `--outdir data/shards/swot_only_ssha/whole_shards`.

- Reshard to pre-batched `.pt` tensors for streaming  
  ```bash
  python scripts/reshard_random_train.py \
      --src_dir data/shards/swot_only/whole_shards \
      --out_dir data/shards/swot_only/batch_shards \
      --seed 0 \
      --batch_size 16384 \
      --batches_per_file 64
  ```
  Repeat for SSHA by pointing `--src_dir/--out_dir` to the SSHA shard folders.

These batch shards plug into experiment YAMLs (e.g., `configs/experiments/swot_ssha_siren.yaml`) via `data.shards_dir` and `data.train_batchshard_dir` with `train_loader: stream`.

## Testing

Run the existing suite (focused on shard indexing and dataset integrity):
```bash
pytest tests/data/test_data_pipeline.py
```
Extend the `tests/` tree alongside any new data transforms, models, or training utilities.

## Configuration & Git Hygiene

- `.gitignore` excludes generated data (`data/`, `experiments/`, Lightning logs, `.pt/.npz/.nc` artefacts) and Python build products.
- Only the documented example configs are tracked: `configs/data/swot_example.yaml`, `configs/data/gulf_config.json`, and `configs/experiments/example_rect_mlp.yaml`. Create site-specific configs by copying these templates; they will remain untracked.
- Keep watermask tiles, shard outputs, and resharded batches under the ignored `data/` tree to avoid large files in version control.

## Additional Notes

- `poseidon.training.SetEpochOnIterable` keeps epoch-aware shuffling in sync for iterable datasets.
- `ShardedDataModule.stats` exposes the context consumed by positional encoders; if you add a new encoder, document any additional keys it requires.
- The download utilities expect `earthaccess.login()` to succeed; configure credentials once before automation (CI jobs should inject them via environment variables, but can be manually input upon job call.).

With these building blocks you can script full SWOT processing pipelines, experiment with alternative positional encoders or regression heads, and run training either headlessly or through the supplied Lightning notebook.

Recent additions include:
- `scripts/run.py` defaulting to the experiment-configured epoch budget unless `--epochs` is supplied.
- Gradient diagnostics inside `LitRegressor`: feature encoder layouts are printed once per run and per-feature-group gradient norms are logged as `grad_norm/<group>` each epoch. Inspect these metrics (CSV logger or TensorBoard) to see which inputs drive the optimisation.

## Grid Loss Search

We routinely sweep positional-encoder hyperparameters to see which combination yields the lowest validation loss before committing to longer runs. The `swot_ssha_rff_mse` experiment showcases this workflow with a grid over spatial/temporal random Fourier feature counts:

1. **Prepare the config** – `configs/experiments/swot_ssha_rff_mse.yaml` enables the physical RFF encoder with min/max time normalisation, annual harmonics, and no target normaliser. Adjust any defaults (learning rate, weight decay, shard paths) before launching the sweep.
2. **Launch the grid** – either call `python scripts/run.py --configs configs/experiments/swot_ssha_rff_mse.yaml --grid model.pe.spatial_features=8,16,32 --grid model.pe.temporal_features=4,8,16` locally or submit `z_sbatch_scripts/run/run_swot_ssha_rff_mse_grid.sbatch` to SLURM. Each Cartesian combination inherits the base config and writes under `experiments/swot_ssha_rff_mse/<grid_id>` with its own Lightning logs and checkpoints.
3. **Rank by loss** – once the jobs finish, run `python scripts/eval_summary.py --experiment-dir experiments/swot_ssha_rff_mse --top 10`. The summary CSV/console output sorts runs by validation loss so you can pick the best-performing feature budget (or spot over/under-fitting trends across the grid).
4. **Promote the winner** – copy the most promising `grid_*` sub-config into a dedicated YAML (or bake the chosen hyperparameters back into `swot_ssha_rff_mse.yaml`) before firing longer training/eval workflows. Rerun the summary after any reruns to keep the loss leaderboard current.

The same pattern works for any hyperparameter subset: add more `--grid key=...` entries, or mix `--set key=val` for single overrides. Because `scripts/run.py` shares context stats across runs, positional encoders always see the correct min/max/mean metadata during the sweep.


## To-Do 
- Update `poseidon.training.SetEpochOnIterable` so shards are handled independently by different workers.
