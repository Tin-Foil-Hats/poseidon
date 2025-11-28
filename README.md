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


## To-Do 
- Update `poseidon.training.SetEpochOnIterable` so shards are handled independently by different workers.
