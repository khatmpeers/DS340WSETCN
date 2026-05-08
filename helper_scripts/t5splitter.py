from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_TOP5 = [
    "traction_tractionForce",
    "odometry_vehicleSpeed",
    "odometry_wheelSpeed_fr",
    "odometry_wheelSpeed_ml",
    "odometry_wheelSpeed_mr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reduced-feature copies of split CSV directories."
    )
    parser.add_argument(
        "--input_root",
        required=True,
        help="Root directory containing train/, val/, and test/ subdirectories.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root directory to write reduced-feature train/, val/, and test/ subdirectories.",
    )
    parser.add_argument(
        "--target_column",
        default="electric_powerDemand",
        help="Target column to preserve.",
    )
    parser.add_argument(
        "--date_column",
        default="time_unix",
        help="Date/time column to preserve.",
    )
    parser.add_argument(
        "--feature_columns",
        nargs="+",
        default=None,
        help="Explicit feature columns to keep. If omitted, uses --shap_csv top-k or built-in default top 5.",
    )
    parser.add_argument(
        "--shap_csv",
        default=None,
        help="Optional path to shap_feature_importance.csv. If provided, top-k features are selected from this file.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top SHAP-ranked features to keep when using --shap_csv.",
    )
    return parser.parse_args()


def resolve_feature_columns(args: argparse.Namespace) -> list[str]:
    if args.feature_columns is not None:
        return list(args.feature_columns)

    if args.shap_csv is not None:
        shap_df = pd.read_csv(args.shap_csv)
        if "feature" not in shap_df.columns:
            raise ValueError("SHAP CSV must contain a 'feature' column.")
        features = shap_df["feature"].head(args.top_k).tolist()
        if len(features) < args.top_k:
            raise ValueError(
                f"Requested top_k={args.top_k}, but SHAP CSV only has {len(features)} rows."
            )
        return features

    return DEFAULT_TOP5.copy()


def process_split(
    split_name: str,
    input_root: Path,
    output_root: Path,
    keep_columns: list[str],
) -> None:
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

        missing = [col for col in keep_columns if col not in df.columns]
        if missing:
            raise ValueError(f"{csv_path.name} is missing required columns: {missing}")

        reduced_df = df[keep_columns].copy()
        reduced_df.to_csv(out_dir / csv_path.name, index=False)

        if i % 10 == 0 or i == len(csv_files):
            print(f"{split_name}: processed {i}/{len(csv_files)}")


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    feature_columns = resolve_feature_columns(args)
    keep_columns = [args.date_column] + feature_columns + [args.target_column]

    print("Keeping columns:")
    for col in keep_columns:
        print(f"  - {col}")

    for split in ["train", "val", "test"]:
        process_split(
            split_name=split,
            input_root=input_root,
            output_root=output_root,
            keep_columns=keep_columns,
        )

    print("\nDone.")
    print(f"Reduced-feature dataset written to: {output_root}")


if __name__ == "__main__":
    main()
