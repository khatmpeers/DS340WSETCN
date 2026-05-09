from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_DATA_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run README steps 6-8 for the DS340W SE-TCN project: "
            "train XGBoost + SHAP, create feature/sparsity dataset variants, "
            "and train SE-TCN on the baseline plus each generated variant."
        )
    )

    parser.add_argument(
        "--data",
        default="reduced_data",
        help=(
            "Path to the already partitioned reduced_data directory from README step 5. "
            "This directory must contain train/, val/, and test/."
        ),
    )
    parser.add_argument(
        "--helper_dir",
        default="helper_scripts",
        help="Directory containing scripts such as feature_splitter.py and sparsity_splitter.py.",
    )
    parser.add_argument("--date_column", default="time_unix", help="Date/time column passed to training scripts.")

    # README step 6 defaults.
    parser.add_argument("--xgb_output_dir", default="outputs_xgboost", help="Output directory for XGBoost + SHAP artifacts.")
    parser.add_argument(
        "--xgb_flat_data_dir",
        default=".pipeline_work/xgboost_flat_data",
        help=(
            "Generated flat CSV directory used only for train_xgboost.py. "
            "This is needed because reduced_data is already partitioned into train/val/test."
        ),
    )
    parser.add_argument("--max_rows_per_trip", type=int, default=2000, help="Row cap per trip for XGBoost and SE-TCN.")
    parser.add_argument("--shap_sample_size", type=int, default=5000, help="Sample size used for SHAP analysis.")

    # README step 7 defaults.
    parser.add_argument("--top5_output_root", default="reduced_data_top5_predictors", help="Output root for top-5 feature dataset.")
    parser.add_argument("--top8_output_root", default="reduced_data_top8_predictors", help="Output root for top-8 feature dataset.")
    parser.add_argument("--sparse2_output_root", default="reduced_data_2x_sparsity", help="Output root for 2x sparsity dataset.")
    parser.add_argument("--sparse5_output_root", default="reduced_data_5x_sparsity", help="Output root for 5x sparsity dataset.")

    # README step 8 defaults.
    parser.add_argument("--outputs_dir", default="outputs", help="Parent output directory for SE-TCN runs.")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs for each SE-TCN run. Reduce this to quickly test functionality.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for each SE-TCN run.")
    parser.add_argument("--lr", default="1e-4", help="Learning rate for each SE-TCN run.")
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device argument for train.py, e.g. cpu, cuda, or mps. Omitted unless provided.",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete generated step 6-8 output directories before running.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )

    return parser.parse_args()


def run_step(name: str, command: list[str], *, dry_run: bool = False) -> None:
    print("\n" + "=" * 80)
    print(f"STEP: {name}")
    print("COMMAND:")
    print(" ".join(command))
    print("=" * 80)

    if dry_run:
        print(f"[DRY RUN] Skipped: {name}")
        return

    result = subprocess.run(command)

    if result.returncode != 0:
        raise SystemExit(
            f"\n[FAIL] Step failed: {name}\n"
            f"Exit code: {result.returncode}\n"
            "Check the console output above for the underlying error."
        )

    print(f"[OK] {name}")


def require_file(path: Path, *, dry_run: bool = False) -> None:
    if dry_run:
        return
    if not path.exists():
        raise SystemExit(f"[FAIL] Expected file was not created: {path}")


def require_dir(path: Path, *, dry_run: bool = False) -> None:
    if dry_run:
        return
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"[FAIL] Expected directory was not created: {path}")


def require_script(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise SystemExit(
            f"[FAIL] Required script not found: {path}\n"
            "Make sure this script is being run from the repository root, or pass --helper_dir."
        )


def require_partitioned_dataset(data_root: Path) -> None:
    if not data_root.exists() or not data_root.is_dir():
        raise SystemExit(
            f"[FAIL] Dataset directory not found: {data_root}\n"
            "Download/place the README step 5 reduced_data folder in the repo, or pass --data."
        )

    missing = [split for split in REQUIRED_DATA_SPLITS if not (data_root / split).is_dir()]
    if missing:
        raise SystemExit(
            f"[FAIL] Dataset root is not partitioned correctly: {data_root}\n"
            f"Missing required split directories: {', '.join(missing)}\n"
            "README steps 6-8 assume reduced_data already contains train/, val/, and test/."
        )

    empty = [split for split in REQUIRED_DATA_SPLITS if not any((data_root / split).glob("*.csv"))]
    if empty:
        raise SystemExit(
            f"[FAIL] Dataset split(s) contain no CSV files under {data_root}: {', '.join(empty)}\n"
            "Expected structure: reduced_data/train/*.csv, reduced_data/val/*.csv, reduced_data/test/*.csv."
        )


def resolve_from_cwd(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else Path.cwd() / path


def prepare_flat_xgb_data_dir(data_root: Path, output_dir: Path, *, dry_run: bool = False) -> Path:
    """
    train_xgboost.py expects --data_path to be a directory containing CSV files directly.
    The README dataset is already partitioned, so this creates a small generated
    directory that exposes all split CSVs in one flat folder. The original
    reduced_data/train, reduced_data/val, and reduced_data/test folders are not changed.
    """
    if dry_run:
        return output_dir

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in REQUIRED_DATA_SPLITS:
        split_dir = data_root / split
        for csv_path in sorted(split_dir.glob("*.csv")):
            destination = output_dir / f"{split}__{csv_path.name}"
            try:
                destination.symlink_to(csv_path.resolve())
            except OSError:
                shutil.copy2(csv_path, destination)

    if not any(output_dir.glob("*.csv")):
        raise SystemExit(f"[FAIL] Could not prepare flat XGBoost input directory: {output_dir}")

    return output_dir


def maybe_clean(paths: list[Path], *, enabled: bool) -> None:
    if not enabled:
        return

    for path in paths:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def load_top_features_from_shap_csv(shap_csv: Path, top_k: int, *, dry_run: bool = False) -> list[str]:
    """
    Read the SHAP feature importance CSV and return the top-k feature names.

    This script intentionally accepts a few likely column names so it can work
    even if train_xgboost.py uses feature, feature_name, or column as the header.
    The CSV is assumed to already be sorted from most important to least important,
    which matches how feature_splitter.py consumes the same file.
    """
    if dry_run and not shap_csv.exists():
        return []

    require_file(shap_csv, dry_run=dry_run)

    with shap_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"[FAIL] SHAP CSV has no header: {shap_csv}")

        normalized = {field.strip().lower(): field for field in reader.fieldnames}
        preferred_names = (
            "feature",
            "feature_name",
            "feature_column",
            "column",
            "predictor",
            "predictor_name",
        )

        feature_field = next(
            (normalized[name] for name in preferred_names if name in normalized),
            None,
        )

        if feature_field is None:
            # Fallback: pick the first column that does not look like an importance value.
            importance_like = ("importance", "shap", "score", "rank", "mean", "value")
            for field in reader.fieldnames:
                lowered = field.strip().lower()
                if not any(token in lowered for token in importance_like):
                    feature_field = field
                    break

        if feature_field is None:
            raise SystemExit(
                f"[FAIL] Could not identify the feature-name column in {shap_csv}.\n"
                f"Available columns: {reader.fieldnames}"
            )

        features: list[str] = []
        for row in reader:
            feature = (row.get(feature_field) or "").strip()
            if feature and feature not in features:
                features.append(feature)
            if len(features) == top_k:
                break

    if len(features) < top_k:
        raise SystemExit(
            f"[FAIL] Expected at least {top_k} features in {shap_csv}, found {len(features)}."
        )

    print(f"Top-{top_k} SHAP features from {shap_csv}:")
    for feature in features:
        print(f"  - {feature}")

    return features


def train_setcn_variant(
    *,
    python: str,
    train_script: Path,
    name: str,
    dataset_root: Path,
    output_dir: Path,
    date_column: str,
    epochs: int,
    batch_size: int,
    lr: str,
    max_rows_per_trip: int,
    feature_columns: list[str] | None,
    device: str | None,
    dry_run: bool,
) -> None:
    for split in REQUIRED_DATA_SPLITS:
        require_dir(dataset_root / split, dry_run=dry_run)

    command = [
        python,
        str(train_script),
        "--model",
        "setcn",
        "--train_dir",
        str(dataset_root / "train"),
        "--val_dir",
        str(dataset_root / "val"),
        "--test_dir",
        str(dataset_root / "test"),
        "--date_column",
        date_column,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--lr",
        str(lr),
        "--max_rows_per_trip",
        str(max_rows_per_trip),
        "--output_dir",
        str(output_dir),
    ]

    if feature_columns:
        command.extend(["--feature_columns", *feature_columns])

    if device:
        command.extend(["--device", device])

    run_step(f"train SE-TCN: {name}", command, dry_run=dry_run)

    require_file(output_dir / "metrics.json", dry_run=dry_run)
    require_file(output_dir / "test_predictions.csv", dry_run=dry_run)
    require_file(output_dir / "run_config.json", dry_run=dry_run)


def main() -> None:
    args = parse_args()

    python = sys.executable
    data_root = resolve_from_cwd(args.data)
    helper_dir = resolve_from_cwd(args.helper_dir)

    train_xgboost_script = Path.cwd() / "train_xgboost.py"
    feature_splitter_script = helper_dir / "feature_splitter.py"
    sparsity_splitter_script = helper_dir / "sparsity_splitter.py"

    require_partitioned_dataset(data_root)
    require_script(train_xgboost_script)
    require_script(feature_splitter_script)
    require_script(sparsity_splitter_script)
    train_script = resolve_from_cwd("train.py")
    require_script(train_script)

    xgb_output_dir = resolve_from_cwd(args.xgb_output_dir)
    xgb_flat_data_dir = resolve_from_cwd(args.xgb_flat_data_dir)
    top5_root = resolve_from_cwd(args.top5_output_root)
    top8_root = resolve_from_cwd(args.top8_output_root)
    sparse2_root = resolve_from_cwd(args.sparse2_output_root)
    sparse5_root = resolve_from_cwd(args.sparse5_output_root)
    outputs_dir = resolve_from_cwd(args.outputs_dir)

    maybe_clean(
        [xgb_output_dir, xgb_flat_data_dir, top5_root, top8_root, sparse2_root, sparse5_root, outputs_dir],
        enabled=args.clean,
    )

    xgb_data_path = prepare_flat_xgb_data_dir(
        data_root,
        xgb_flat_data_dir,
        dry_run=args.dry_run,
    )

    run_step(
        "README step 6: train surrogate XGBoost model + SHAP analysis",
        [
            python,
            str(train_xgboost_script),
            "--data_path",
            str(xgb_data_path),
            "--date_column",
            args.date_column,
            "--max_rows_per_trip",
            str(args.max_rows_per_trip),
            "--shap_sample_size",
            str(args.shap_sample_size),
            "--output_dir",
            str(xgb_output_dir),
        ],
        dry_run=args.dry_run,
    )

    shap_csv = xgb_output_dir / "shap_feature_importance.csv"
    require_file(shap_csv, dry_run=args.dry_run)

    variant_steps = [
        (
            "README step 7: produce top-5 predictors dataset",
            [
                python,
                str(feature_splitter_script),
                "--input_root",
                str(data_root),
                "--output_root",
                str(top5_root),
                "--shap_csv",
                str(shap_csv),
                "--top_k",
                "5",
            ],
            top5_root,
        ),
        (
            "README step 7: produce top-8 predictors dataset",
            [
                python,
                str(feature_splitter_script),
                "--input_root",
                str(data_root),
                "--output_root",
                str(top8_root),
                "--shap_csv",
                str(shap_csv),
                "--top_k",
                "8",
            ],
            top8_root,
        ),
        (
            "README step 7: produce 2x sparsity dataset",
            [
                python,
                str(sparsity_splitter_script),
                "--input_root",
                str(data_root),
                "--output_root",
                str(sparse2_root),
                "--step",
                "2",
            ],
            sparse2_root,
        ),
        (
            "README step 7: produce 5x sparsity dataset",
            [
                python,
                str(sparsity_splitter_script),
                "--input_root",
                str(data_root),
                "--output_root",
                str(sparse5_root),
                "--step",
                "5",
            ],
            sparse5_root,
        ),
    ]

    for name, command, output_root in variant_steps:
        run_step(name, command, dry_run=args.dry_run)
        for split in REQUIRED_DATA_SPLITS:
            require_dir(output_root / split, dry_run=args.dry_run)

    top5_features = load_top_features_from_shap_csv(shap_csv, 5, dry_run=args.dry_run)
    top8_features = load_top_features_from_shap_csv(shap_csv, 8, dry_run=args.dry_run)

    train_variants = [
        ("baseline", data_root, outputs_dir / f"baseline_setcn_{args.epochs}ep", None),
        ("2x sparsity", sparse2_root, outputs_dir / f"2x_sparsity_setcn_{args.epochs}ep", None),
        ("5x sparsity", sparse5_root, outputs_dir / f"5x_sparsity_setcn_{args.epochs}ep", None),
        ("top-5 predictors", top5_root, outputs_dir / f"top5_setcn_{args.epochs}ep", top5_features),
        ("top-8 predictors", top8_root, outputs_dir / f"top8_setcn_{args.epochs}ep", top8_features),
    ]

    for name, dataset_root, output_dir, feature_columns in train_variants:
        train_setcn_variant(
            python=python,
            train_script=train_script,
            name=name,
            dataset_root=dataset_root,
            output_dir=output_dir,
            date_column=args.date_column,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_rows_per_trip=args.max_rows_per_trip,
            feature_columns=feature_columns,
            device=args.device,
            dry_run=args.dry_run,
        )

    print("\n" + "=" * 80)
    print("[SUCCESS] README steps 6-8 completed.")
    print("Generated/used:")
    print(f"  XGBoost flat CSV input: {xgb_flat_data_dir}")
    print(f"  SHAP output: {xgb_output_dir}")
    print(f"  Top-5 dataset: {top5_root}")
    print(f"  Top-8 dataset: {top8_root}")
    print(f"  2x sparsity dataset: {sparse2_root}")
    print(f"  5x sparsity dataset: {sparse5_root}")
    print(f"  SE-TCN outputs: {outputs_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
