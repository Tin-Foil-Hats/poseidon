                      
"""Shuffle training SWOT shards into batched Torch tensors."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from poseidon.data.shards import reshard_random_train


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly re-shard training samples into batch tensors.")
    parser.add_argument("--src_dir", type=Path, required=True, help="Directory containing input .npz shards")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory for output batch shard .pt files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=16384, help="Samples per batch inside each .pt shard")
    parser.add_argument("--batches_per_file", type=int, default=64, help="Number of batches stored per output file")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio for original shards")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Test split ratio for original shards")
    parser.add_argument(
        "--clip_quantiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Drop samples with targets outside these quantile bounds (values in [0,1]).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    result = reshard_random_train(
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        batches_per_file=args.batches_per_file,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        clip_quantiles=tuple(args.clip_quantiles) if args.clip_quantiles else None,
    )

    print(
        "loaded={samples} batches={batches} batch_size={bs} files={files} dropped={dropped}".format(
            samples=result.samples_loaded,
            batches=result.batches_written,
            bs=result.batch_size,
            files=len(result.written),
            dropped=result.dropped_samples,
        )
    )

    if result.clipped_samples:
        lo, hi = result.clip_bounds if result.clip_bounds else (None, None)
        print(f"clipped={result.clipped_samples} clip_bounds=({lo}, {hi})")

    if result.missing_shards:
        print(f"warning: skipped {result.missing_shards} missing shard(s)")

    for split, groups in result.split_groups.items():
        label = f"{split}: {len(groups)} group(s)"
        if groups:
            pairs = ", ".join(f"c{cyc:03d}/p{pas:03d}" for cyc, pas in groups)
            label += f" -> {pairs}"
        print(label)

    for path in result.written:
        print(f"wrote {Path(path).name}")


if __name__ == "__main__":
    main()
