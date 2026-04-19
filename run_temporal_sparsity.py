from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from model import DILATIONS, HIDDEN_CHANNELS, KERNEL_SIZE, WINDOW_SIZE, build_model
from train import (
    DEFAULT_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_YEARS,
    TRAIN_YEARS,
    build_dataloader,
    collect_predictions,
    compute_metrics,
    create_sliding_windows,
    fit_feature_scaler,
    resolve_device,
    set_seed,
    train_one_epoch,
)
from train_xgboost import load_trip_dataframes, split_trips_by_year


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SE-TCN temporal sparsity experiments by downsampling before window creation."
    )
    parser.add_argument("--data_path", required=True, help="Path to the DTB CSV data file.")
    parser.add_argument(
        "--model",
        default="setcn",
        choices=["tcn", "tcn_r1", "tcn_r2", "tcn_r3", "setcn"],
        help="Model variant to train for each sampling interval.",
    )
    parser.add_argument(
        "--sampling_intervals",
        nargs="+",
        required=True,
        type=int,
        help="Sampling intervals to evaluate, for example: 1 2 5 10",
    )
    parser.add_argument(
        "--feature_columns",
        nargs="+",
        default=None,
        help="Optional feature override. Defaults to the paper-faithful retained variables.",
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
        default="outputs_temporal_sparsity",
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


def resolve_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "results": Path(args.results_path) if args.results_path else output_dir / "temporal_sparsity_results.csv",
        "run_config": Path(args.run_config_path) if args.run_config_path else output_dir / "run_config.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def downsample_dataframe(dataframe: pd.DataFrame, interval: int) -> pd.DataFrame:
    if interval <= 0:
        raise ValueError("Sampling intervals must be positive integers.")
    return dataframe.iloc[::interval].reset_index(drop=True)


def combine_downsampled_trips(
    trips: list[tuple[Path, pd.DataFrame]],
    interval: int,
) -> pd.DataFrame:
    downsampled_trips = [downsample_dataframe(trip_df, interval) for _, trip_df in trips]
    non_empty_trips = [trip_df for trip_df in downsampled_trips if not trip_df.empty]
    if not non_empty_trips:
        raise ValueError(
            f"No rows remain after trip-wise downsampling with interval={interval}."
        )
    return pd.concat(non_empty_trips, ignore_index=True)


def train_and_evaluate_interval(
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


def build_run_config(
    args: argparse.Namespace,
    feature_columns: list[str],
    date_column: str,
) -> dict:
    return {
        "data_path": args.data_path,
        "model": args.model,
        "sampling_intervals": args.sampling_intervals,
        "feature_columns": feature_columns,
        "window_size": WINDOW_SIZE,
        "dilations": list(DILATIONS),
        "kernel_size": KERNEL_SIZE,
        "hidden_channels": HIDDEN_CHANNELS,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "target_column": args.target_column,
        "date_column": date_column,
        "train_years": list(TRAIN_YEARS),
        "test_years": list(TEST_YEARS),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_paths = resolve_output_paths(args)
    device = resolve_device(args.device)

    requested_features = args.feature_columns if args.feature_columns is not None else DEFAULT_FEATURE_COLUMNS
    trips, date_column, feature_columns = load_trip_dataframes(
        data_path=args.data_path,
        target_column=args.target_column,
        date_column=args.date_column,
        feature_columns=requested_features,
    )
    train_trips, test_trips = split_trips_by_year(trips, date_column)

    results = []
    print(f"Using date column: {date_column}")
    print(f"Model variant: {args.model}")
    print(f"Using feature columns: {feature_columns}")
    print(f"Training trips available: {len(train_trips)}")
    print(f"Testing trips available: {len(test_trips)}")

    for interval in args.sampling_intervals:
        downsampled_train_df = combine_downsampled_trips(train_trips, interval)
        downsampled_test_df = combine_downsampled_trips(test_trips, interval)

        print(
            f"Running sampling_interval={interval} "
            f"(train_rows={len(downsampled_train_df)}, test_rows={len(downsampled_test_df)})"
        )
        metrics = train_and_evaluate_interval(
            downsampled_train_df,
            downsampled_test_df,
            feature_columns,
            args,
            device,
        )

        results.append(
            {
                "sampling_interval": interval,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_paths["results"], index=False)
    save_json(output_paths["run_config"], build_run_config(args, feature_columns, date_column))

    print(f"Saved temporal sparsity results to {output_paths['results']}")
    print(f"Saved run config to {output_paths['run_config']}")


if __name__ == "__main__":
    main()
