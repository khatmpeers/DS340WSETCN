from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from model import DILATIONS, HIDDEN_CHANNELS, KERNEL_SIZE, WINDOW_SIZE, build_model
from train import (
    TARGET_COLUMN,
    TRAIN_YEARS,
    TEST_YEARS,
    build_dataloader,
    collect_predictions,
    compute_metrics,
    create_sliding_windows,
    fit_feature_scaler,
    load_dataframe,
    resolve_device,
    set_seed,
    split_by_year,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SE-TCN feature-reduction experiments from aggregated SHAP rankings."
    )
    parser.add_argument(
        "--ranking_csv",
        required=True,
        help="Path to the aggregated feature-level SHAP importance CSV.",
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to the DTB CSV data file.",
    )
    parser.add_argument(
        "--model",
        default="setcn",
        choices=["tcn", "tcn_r1", "tcn_r2", "tcn_r3", "setcn"],
        help="Model variant to train for each subset size.",
    )
    parser.add_argument(
        "--subset_sizes",
        nargs="+",
        required=True,
        type=int,
        help="Feature subset sizes to evaluate, for example: 5 8 10 12 15 18",
    )
    parser.add_argument(
        "--target_column",
        default=TARGET_COLUMN,
        help="Target column to predict. Defaults to electric_powerDemand.",
    )
    parser.add_argument(
        "--date_column",
        default=None,
        help="Optional datetime column. If omitted, the script will try to detect one.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability used inside TCN layers.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use, for example 'cpu', 'cuda', or 'mps'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs_feature_reduction",
        help="Directory for the results CSV and run config JSON.",
    )
    parser.add_argument(
        "--results_path",
        default=None,
        help="Optional override for the results CSV path.",
    )
    parser.add_argument(
        "--run_config_path",
        default=None,
        help="Optional override for the run config JSON path.",
    )
    return parser.parse_args()


def load_ranked_features(ranking_csv: str) -> list[str]:
    ranking_df = pd.read_csv(ranking_csv)
    if "feature" in ranking_df.columns:
        ranked_features = ranking_df["feature"].astype(str).tolist()
    else:
        first_column = ranking_df.columns[0]
        ranked_features = ranking_df[first_column].astype(str).tolist()

    ranked_features = [feature for feature in ranked_features if feature]
    if not ranked_features:
        raise ValueError(f"No ranked features found in {ranking_csv}.")
    return ranked_features


def resolve_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "results": Path(args.results_path) if args.results_path else output_dir / "feature_reduction_results.csv",
        "run_config": Path(args.run_config_path) if args.run_config_path else output_dir / "run_config.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def train_and_evaluate_subset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    scaler = fit_feature_scaler(train_df, feature_columns)
    scaled_train_df = scaler.transform(train_df, feature_columns)
    scaled_test_df = scaler.transform(test_df, feature_columns)

    train_data = create_sliding_windows(scaled_train_df, feature_columns, args.target_column)
    test_data = create_sliding_windows(scaled_test_df, feature_columns, args.target_column)

    train_loader = build_dataloader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = build_dataloader(test_data, batch_size=args.batch_size, shuffle=False)

    model = build_model(
        args.model,
        input_channels=len(feature_columns),
        dropout=args.dropout,
    ).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for _ in range(args.epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

    y_true, y_pred = collect_predictions(model, test_loader, device)
    return compute_metrics(y_true, y_pred)


def build_run_config(args: argparse.Namespace, available_ranked_features: list[str]) -> dict:
    return {
        "ranking_csv": args.ranking_csv,
        "data_path": args.data_path,
        "model": args.model,
        "subset_sizes": args.subset_sizes,
        "window_size": WINDOW_SIZE,
        "dilations": list(DILATIONS),
        "kernel_size": KERNEL_SIZE,
        "hidden_channels": HIDDEN_CHANNELS,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "target_column": args.target_column,
        "date_column": args.date_column,
        "train_years": list(TRAIN_YEARS),
        "test_years": list(TEST_YEARS),
        "ranked_features_available": available_ranked_features,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_paths = resolve_output_paths(args)
    device = resolve_device(args.device)

    ranked_features = load_ranked_features(args.ranking_csv)
    dataframe, date_column, available_features = load_dataframe(
        data_path=args.data_path,
        target_column=args.target_column,
        date_column=args.date_column,
        feature_columns=ranked_features,
    )
    train_df, test_df = split_by_year(dataframe, date_column)

    results = []
    print(f"Using date column: {date_column}")
    print(f"Model variant: {args.model}")

    for subset_size in args.subset_sizes:
        if subset_size <= 0:
            raise ValueError("All subset sizes must be positive integers.")
        if subset_size > len(available_features):
            raise ValueError(
                f"Requested subset size {subset_size}, but only {len(available_features)} ranked features are available."
            )

        subset_features = available_features[:subset_size]
        print(f"Running subset_size={subset_size} with features: {subset_features}")
        metrics = train_and_evaluate_subset(train_df, test_df, subset_features, args, device)

        results.append(
            {
                "subset_size": subset_size,
                "features_used": ",".join(subset_features),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_paths["results"], index=False)
    save_json(output_paths["run_config"], build_run_config(args, available_features))

    print(f"Saved feature reduction results to {output_paths['results']}")
    print(f"Saved run config to {output_paths['run_config']}")


if __name__ == "__main__":
    main()
