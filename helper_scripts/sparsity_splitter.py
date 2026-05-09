from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create temporally sparse copies of split CSV directories by keeping every Nth row."
    )
    parser.add_argument(
        "--input_root",
        required=True,
        help="Root directory containing train/, val/, and test/ subdirectories.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root directory to write sparse train/, val/, and test/ subdirectories.",
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Temporal sparsity factor. For example, 2 keeps every 2nd row; 5 keeps every 5th row.",
    )
    return parser.parse_args()


def process_split(split_name: str, input_root: Path, output_root: Path, step: int) -> None:
    in_dir = input_root / split_name
    out_dir = output_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise ValueError(f"Missing input split directory: {in_dir}")

    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {in_dir}")

    print(f"{split_name}: {len(csv_files)} files")

    for i, csv_path in enumerate(csv_files, start=1):
        df = pd.read_csv(csv_path)
        sparse_df = df.iloc[::step].copy()
        sparse_df.to_csv(out_dir / csv_path.name, index=False)

        if i % 10 == 0 or i == len(csv_files):
            print(f"{split_name}: processed {i}/{len(csv_files)}")


def main() -> None:
    args = parse_args()
    if args.step <= 0:
        raise ValueError("--step must be a positive integer.")

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    print(f"Creating {args.step}x sparse dataset")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")

    for split in ["train", "val", "test"]:
        process_split(split, input_root, output_root, args.step)

    print("\nDone.")
    print(f"Sparse dataset written to: {output_root}")


if __name__ == "__main__":
    main()
