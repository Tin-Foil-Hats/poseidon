# Poseidon Function Index

This document enumerates top-level functions and class methods within `src/poseidon`.
Generated automatically via AST parsing.

## __init__.py

*(no top-level functions or classes)*

## constants.py

**Classes**
- TidalConstituent
  - from_hours

## data/__init__.py

*(no top-level functions or classes)*

## data/dataset.py

**Classes**
- ShardedFlatDataset
  - __init__
  - __len__
  - _load_shard_cached
  - __getitem__
- BatchShardIterable
  - __init__
  - __iter__
- PassShardDataset
  - __init__
  - __len__
  - __getitem__

## data/schema.py

**Functions**
- validate_shard
- load_shard
- save_shard
- shard_length

## data/shards.py

**Functions**
- parse_cycle_pass_from_name
- scan_shards
- save_index
- load_index
- split_by_group
- split_by_cycle_pass
- _resolve_shard_path
- reshard_random_train

**Classes**
- ReshardResult

## data/sources/__init__.py

*(no top-level functions or classes)*

## data/sources/base.py

**Functions**
- register_source
- get_source

**Classes**
- DataSource
  - download
  - preprocess
  - build_shards

## data/sources/coops/__init__.py

*(no top-level functions or classes)*

## data/sources/coops/download.py

**Functions**
- main

**Classes**
- NOAAStationFetcher
  - __init__
  - fetch_stations
  - filter_by_bbox
  - _get_station_details
  - _station_meets_criteria
  - run
  - to_dataframe
  - save_txt
  - display

## data/sources/sentinel6/__init__.py

*(no top-level functions or classes)*

## data/sources/sentinel6/download.py

**Functions**
- parse_name
- load_config
- search_all
- best_per_start
- reconcile_strict
- main

## data/sources/swot/__init__.py

**Classes**
- SWOTSource
  - download
  - preprocess
  - build_shards

## data/sources/swot/download.py

**Functions**
- _require_api
- ensure_login
- _with_source
- search_swot_granules
- _native_id
- _match_name
- _minor_rank
- _major_rank
- _crid_score
- filter_passes
- select_best_by_timestamp
- _scan_local_granules
- _build_download_actions
- apply_download_plan
- run_swot_download
- _coerce_bbox
- _coerce_passes
- main

**Classes**
- SWOTQuery
  - __post_init__
- SWOTDownloadPlan
- SWOTDownloadResult
  - purged
- SWOTDownloadConfig
  - from_mapping
  - to_query
  - run

## data/sources/swot/preprocess.py

**Functions**
- query_landmask_values_at_swot
- parse_filename_seasonality
- is_within_bounding_box
- filter_filenames_seasonality
- calculate_window_seasonality
- load_multiple_seasonality_files

## data/sources/swot/shard_builder.py

**Functions**
- parse_cycle_pass
- _get_first
- to_seconds_like
- _load_measurement
- build_shards

## data/sources/watermask/__init__.py

*(no top-level functions or classes)*

## data/sources/watermask/download.py

**Functions**
- main

## data/transforms.py

**Classes**
- TargetNormalizer
  - active
  - transform
  - inverse
  - to_dict
  - from_dict
  - fit

## models/__init__.py

*(no top-level functions or classes)*

## models/losses/__init__.py

**Functions**
- discover_losses
- list_registered_losses
- build_loss

## models/losses/base.py

**Functions**
- _register_key
- register_loss
- get_loss
- available_losses
- build_loss

## models/losses/huber.py

**Functions**
- huber

## models/losses/losses.py

*(no top-level functions or classes)*

## models/losses/mae.py

**Functions**
- mae

## models/losses/mse.py

**Functions**
- mse

## models/nets/__init__.py

**Functions**
- discover_nets
- list_registered_nets
- build_net

## models/nets/base.py

**Functions**
- _register_key
- register_net
- get_net_class
- available_nets
- build_net

**Classes**
- NetBase
  - out_dim

## models/nets/deep_random_features.py

**Functions**
- _make_rff_layer
- _spherical_to_cartesian
- _expand

**Classes**
- _RFFLayer
  - __init__
  - reset_parameters
  - forward
  - _sample_weight
- _GaussianRFFLayer
  - __init__
  - _sample_weight
- _MaternRFFLayer
  - __init__
  - _sample_weight
- RandomPhaseFeatureMap
  - __init__
  - _polyval
  - _addition_constant
  - __call__
- MaternRandomPhaseS2RFFLayer
  - __init__
  - forward
- SumFeatures
  - __init__
  - forward
- DeepRandomFeaturesNet
  - __init__
  - forward
  - out_dim

## models/nets/dual_branch.py

**Functions**
- _normalize_cfg

**Classes**
- DualBranchNet
  - __init__
  - bind_encoder_layout
  - _apply_layout
  - _branch_output_dim
  - _build_branch
  - _build_fusion_head
  - _prepare_output
  - forward

## models/nets/fcnet.py

**Classes**
- _ResidualBlock
  - __init__
  - forward
- FCNet
  - __init__
  - forward

## models/nets/mlp.py

**Classes**
- MLP
  - __init__
  - forward

## models/nets/nets.py

*(no top-level functions or classes)*

## models/nets/resmlp.py

**Classes**
- ResBlock
  - __init__
  - forward
- ResMLP
  - __init__
  - forward

## models/nets/siren.py

**Classes**
- _Sine
  - __init__
  - forward
- _SirenLayer
  - __init__
  - _init_weights
  - forward
- SIREN
  - __init__
  - forward

## models/nets/siren_stable.py

**Classes**
- SIRENStable
  - __init__
  - forward

## models/pe/__init__.py

**Functions**
- discover_pes
- list_registered_pes
- build_pe

## models/pe/base.py

**Classes**
- PEBase
  - feat_dim
  - feature_layout

## models/pe/cossin_simple.py

**Classes**
- PECosSinSimple
  - __init__
  - bind_context
  - feat_dim
  - forward

## models/pe/fourier.py

**Classes**
- PEFourier
  - __init__
  - bind_context
  - feat_dim
  - forward
  - feature_layout
  - _update_layout_cache

## models/pe/identity.py

**Classes**
- PEIdentity
  - __init__
  - feat_dim
  - forward
  - feature_layout

## models/pe/raw_xyz.py

**Classes**
- PERawXYZ
  - __init__
  - feat_dim
  - forward

## models/pe/rectangular_baseline.py

**Classes**
- PERectangularBaseline
  - __init__
  - bind_context
  - feat_dim
  - forward
  - _encode
  - feature_layout
  - _update_layout_cache

## models/pe/registry.py

**Functions**
- _register_key
- register_pe
- get_pe_class
- available_pes
- load_class
- build_pe

## models/pe/rff.py

**Classes**
- PERFF
  - __init__
  - bind_context
  - feat_dim
  - forward

## models/pe/rff_ard_bands.py

**Classes**
- PERFFARDBands
  - __init__
  - bind_context
  - feat_dim
  - forward

## models/pe/rff_with_tides.py

**Functions**
- _omega_rad_per_sec

**Classes**
- PERFFWithTides
  - __init__
  - bind_context
  - feat_dim
  - _time_channel
  - forward
  - feature_layout

## models/pe/satclip_spherical.py

**Functions**
- _associated_legendre_polynomial
- _sh_renorm
- _real_spherical_harmonic

**Classes**
- SatclipSphericalHarmonics
  - __init__
  - feat_dim
  - feature_layout
  - forward

## models/pe/time_only.py

**Classes**
- PETimeOnly
  - __init__
  - bind_context
  - feat_dim
  - forward
  - feature_layout
  - _update_layout_cache

## models/pe/time_with_harmonics.py

**Functions**
- _to_periods

**Classes**
- PETimeWithHarmonics
  - __init__
  - bind_context
  - feat_dim
  - forward
  - feature_layout
  - _update_layout_cache

## models/pe/time_with_raw.py

**Classes**
- PETimeWithRaw
  - __init__
  - bind_context
  - feat_dim
  - forward
  - feature_layout
  - _update_layout_cache

## models/pe/utils.py

**Functions**
- deg2rad
- wrap_lon_pi
- wrap_lon_pi_float
- normalize_lon_bounds
- time_normalize

**Classes**
- FeatureBuilder
  - __init__
  - _register
  - add
  - add_space
  - add_time
  - add_other
  - build
  - layout

## models/poseidon_model.py

**Functions**
- build_model

**Classes**
- LocEncModel
  - __init__
  - forward
  - feature_layout

## training/__init__.py

*(no top-level functions or classes)*

## training/callbacks.py

**Classes**
- SetEpochOnIterable
  - on_train_epoch_start

## training/lightning_datamodule.py

**Functions**
- create_datamodule_from_config

**Classes**
- _BatchShardIterable
  - __init__
  - set_epoch
  - _rng
  - _load_or_build_index
  - __len__
  - __iter__
  - _load_file
- ShardedDataModule
  - __init__
  - _resolve
  - _abs_split
  - _scan_bbox_npz
  - _scan_time_npz
  - _scan_target_npz
  - _scan_stats_pt_train
  - setup
  - _loader
  - _pass_collate
  - train_dataloader
  - val_dataloader
  - test_dataloader

## training/lit_module.py

**Classes**
- LitRegressor
  - __init__
  - training_step
  - on_fit_start
  - validation_step
  - test_step
  - on_after_backward
  - on_train_epoch_end
  - _prepare_grad_monitor
  - _find_first_linear
  - configure_optimizers

## training/perf_monitor.py

**Functions**
- _now_s
- _safe_ratio

**Classes**
- GPUSampler
  - __init__
  - run
  - stop
  - drain
- IOSampler
  - __init__
  - run
  - stop
  - drain
- BatchTiming
- PerfCallback
  - __init__
  - on_fit_start
  - on_fit_end
  - on_before_batch_transfer
  - on_after_batch_transfer
  - on_train_batch_start
  - on_train_batch_end
  - _flush_csv
- TorchOperatorProfiler
  - __init__
  - __enter__
  - __exit__

