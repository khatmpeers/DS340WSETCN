from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import DILATIONS, HIDDEN_CHANNELS, KERNEL_SIZE, WINDOW_SIZE, build_model


TARGET_COLUMN = "electric_powerDemand"
COMMON_DATE_COLUMNS = ("date", "datetime", "timestamp", "time", "time_unix")
DEFAULT_FEATURE_COLUMNS = [
    "gnss_latitude",
    "gnss_longitude",
    "odometry_articulationAngle",
    "odometry_steeringAngle",
    "odometry_vehicleSpeed",
    "odometry_wheelSpeed_fl",
    "odometry_wheelSpeed_fr",
    "odometry_wheelSpeed_ml",
    "odometry_wheelSpeed_mr",
    "odometry_wheelSpeed_rl",
    "odometry_wheelSpeed_rr",
    "status_doorIsOpen",
    "status_gridIsAvailable",
    "status_haltBrakeIsActive",
    "status_parkBrakeIsActive",
    "temperature_ambient",
    "traction_brakePressure",
    "traction_tractionForce",
]


@dataclass
class WindowedSeries:
    inputs: np.ndarray
    targets: np.ndarray


@dataclass
class StandardScaler:
    """Feature-only standardization fitted on the training split."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, dataframe: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        transformed = dataframe.copy()
        values = transformed[feature_columns].to_numpy(dtype=np.float32)
        transformed.loc[:, feature_columns] = (values - self.mean) / self.std
        return transformed


class SlidingWindowDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SE-TCN variants for DTB energy consumption prediction."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["tcn", "tcn_r1", "tcn_r2", "tcn_r3", "setcn"],
        help="Model variant to train.",
    )
    parser.add_argument(
        "--train_dir",
        required=True,
        help="Directory containing training trip CSV files.",
    )
    parser.add_argument(
        "--val_dir",
        default=None,
        help="Optional directory containing validation trip CSV files.",
    )
    parser.add_argument(
        "--test_dir",
        required=True,
        help="Directory containing testing trip CSV files.",
    )
    parser.add_argument(
        "--target_column",
        default=TARGET_COLUMN,
        help="Target column to predict. Defaults to electric_powerDemand.",
    )
    parser.add_argument(
        "--feature_columns",
        nargs="+",
        default=None,
        help="Optional feature override. Defaults to the paper-faithful retained variables.",
    )
    parser.add_argument(
        "--date_column",
        default=None,
        help="Optional datetime column. If omitted, the script will try to detect one.",
    )
    parser.add_argument(
        "--max_rows_per_trip",
        type=int,
        default=None,
        help="Optional cap on the number of rows to keep from each trip after sorting and filtering.",
    )
    parser.add_argument("--epochs", required=True, type=int, help="Number of training epochs.")
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
        "--log_every_batches",
        type=int,
        default=1000,
        help="Optional batch interval for intra-epoch progress logging.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory for predictions, metrics, run config, and optional checkpoint.",
    )
    parser.add_argument(
        "--predictions_path",
        default=None,
        help="Optional override for the predictions CSV path.",
    )
    parser.add_argument(
        "--metrics_path",
        default=None,
        help="Optional override for the metrics JSON path.",
    )
    parser.add_argument(
        "--run_config_path",
        default=None,
        help="Optional override for the run config JSON path.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Optional override for the best validation checkpoint path.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_date_column(dataframe: pd.DataFrame) -> str:
    """Infer the datetime column using common names, then full-column parsing."""

    lowered = {column.lower(): column for column in dataframe.columns}
    for candidate in COMMON_DATE_COLUMNS:
        if candidate in lowered:
            return lowered[candidate]

    for column in dataframe.columns:
        parsed = pd.to_datetime(dataframe[column], errors="coerce")
        if parsed.notna().all():
            return column

    raise ValueError("Could not infer a datetime column. Pass --date_column explicitly.")


def select_feature_columns(
    dataframe: pd.DataFrame,
    target_column: str,
    date_column: str,
    feature_columns: list[str] | None,
) -> list[str]:
    """Use the paper-faithful defaults unless the user explicitly overrides them."""

    if target_column not in dataframe.columns:
        raise ValueError(f"Column '{target_column}' not found in the input CSV.")

    if date_column not in dataframe.columns:
        raise ValueError(f"Column '{date_column}' not found in the input CSV.")

    selected = feature_columns if feature_columns is not None else DEFAULT_FEATURE_COLUMNS
    missing = [column for column in selected if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Feature columns not found in the input CSV: {missing}")

    return list(selected)


def load_trip_dataframe(
    csv_path: Path,
    target_column: str,
    date_column: str | None,
    feature_columns: list[str] | None,
    max_rows_per_trip: int | None,
) -> tuple[pd.DataFrame, str, list[str]]:
    dataframe = pd.read_csv(csv_path)
    resolved_date_column = date_column or infer_date_column(dataframe)
    dataframe = dataframe.copy()

    if resolved_date_column == "time_unix":
        dataframe[resolved_date_column] = pd.to_datetime(
            dataframe[resolved_date_column],
            unit="s",
            errors="coerce",
        )
    else:
        dataframe[resolved_date_column] = pd.to_datetime(
            dataframe[resolved_date_column],
            errors="coerce",
        )

    dataframe = dataframe.dropna(subset=[resolved_date_column]).sort_values(resolved_date_column)

    selected_features = select_feature_columns(
        dataframe=dataframe,
        target_column=target_column,
        date_column=resolved_date_column,
        feature_columns=feature_columns,
    )

    required_columns = selected_features + [target_column]
    dataframe = dataframe.dropna(subset=required_columns).reset_index(drop=True)
    if max_rows_per_trip is not None:
        dataframe = dataframe.iloc[:max_rows_per_trip].copy()
    return dataframe, resolved_date_column, selected_features


def load_trips_from_directory(
    directory_path: str,
    target_column: str,
    date_column: str | None,
    feature_columns: list[str] | None,
    max_rows_per_trip: int | None,
) -> tuple[list[tuple[Path, pd.DataFrame]], str, list[str]]:
    directory = Path(directory_path)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Expected a directory, got: {directory}")

    csv_paths = sorted(directory.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"No CSV files found in directory: {directory}")

    trips: list[tuple[Path, pd.DataFrame]] = []
    resolved_date_column: str | None = None
    selected_features: list[str] | None = None

    for csv_path in csv_paths:
        trip_df, trip_date_column, trip_features = load_trip_dataframe(
            csv_path=csv_path,
            target_column=target_column,
            date_column=date_column,
            feature_columns=feature_columns,
            max_rows_per_trip=max_rows_per_trip,
        )

        if resolved_date_column is None:
            resolved_date_column = trip_date_column
        elif resolved_date_column != trip_date_column:
            raise ValueError(
                f"Inconsistent date columns detected: '{resolved_date_column}' and '{trip_date_column}'."
            )

        if selected_features is None:
            selected_features = trip_features
        elif selected_features != trip_features:
            raise ValueError("Inconsistent feature columns detected across trip CSV files.")

        trips.append((csv_path, trip_df))

    return trips, resolved_date_column or "", selected_features or []


def create_sliding_windows(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    window_size: int = WINDOW_SIZE,
) -> WindowedSeries:
    """Build multivariate windows and next-step targets."""

    features = dataframe[feature_columns].to_numpy(dtype=np.float32)
    targets = dataframe[target_column].to_numpy(dtype=np.float32)

    if len(dataframe) <= window_size:
        raise ValueError(
            f"Need more than {window_size} rows to build sliding windows, got {len(dataframe)}."
        )

    window_inputs = []
    window_targets = []

    for start_index in range(len(dataframe) - window_size):
        end_index = start_index + window_size
        window_inputs.append(features[start_index:end_index])
        window_targets.append(targets[end_index])

    return WindowedSeries(
        inputs=np.asarray(window_inputs, dtype=np.float32),
        targets=np.asarray(window_targets, dtype=np.float32),
    )


def create_sliding_windows_from_trips(
    trips: list[tuple[Path, pd.DataFrame]],
    feature_columns: list[str],
    target_column: str,
    split_name: str,
    window_size: int = WINDOW_SIZE,
) -> WindowedSeries:
    all_inputs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for trip_index, (_, trip_df) in enumerate(trips, start=1):
        if len(trip_df) <= window_size:
            continue

        trip_windows = create_sliding_windows(
            dataframe=trip_df,
            feature_columns=feature_columns,
            target_column=target_column,
            window_size=window_size,
        )
        all_inputs.append(trip_windows.inputs)
        all_targets.append(trip_windows.targets)

        cumulative_windows = int(sum(array.shape[0] for array in all_inputs))
        print(
            f"{split_name}: processed trips {trip_index}/{len(trips)} - "
            f"cumulative windows: {cumulative_windows}"
        )

    if not all_inputs:
        raise ValueError(
            f"No eligible sliding windows were produced for split '{split_name}' with window_size={window_size}."
        )

    return WindowedSeries(
        inputs=np.concatenate(all_inputs, axis=0),
        targets=np.concatenate(all_targets, axis=0),
    )


def fit_feature_scaler(train_df: pd.DataFrame, feature_columns: list[str]) -> StandardScaler:
    """Fit standardization parameters on training features only."""

    values = train_df[feature_columns].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return StandardScaler(mean=mean.astype(np.float32), std=std.astype(np.float32))


def build_dataloader(data: WindowedSeries, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = SlidingWindowDataset(data.inputs, data.targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def move_inputs_to_conv1d(inputs: torch.Tensor) -> torch.Tensor:
    return inputs.transpose(1, 2)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_batches = len(loader)

    for batch_index, (inputs, targets) in enumerate(loader, start=1):
        inputs = move_inputs_to_conv1d(inputs).to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if log_every is not None and log_every > 0 and batch_index % log_every == 0:
            running_mae = total_loss / total_samples
            print(
                f"  batch {batch_index}/{total_batches} - "
                f"running_train_mae: {running_mae:.6f}"
            )

    return total_loss / total_samples


def evaluate_mae(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = move_inputs_to_conv1d(inputs).to(device)
            targets = targets.to(device)

            predictions = model(inputs)
            loss = criterion(predictions, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true_values = []
    predicted_values = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = move_inputs_to_conv1d(inputs).to(device)
            predictions = model(inputs)

            true_values.append(targets.numpy())
            predicted_values.append(predictions.cpu().numpy())

    y_true = np.concatenate(true_values)
    y_pred = np.concatenate(predicted_values)
    return y_true, y_pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    nonzero_mask = y_true != 0
    if np.any(nonzero_mask):
        mape = float(
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]))
            * 100.0
        )
    else:
        mape = float("nan")

    return {"mae": mae, "rmse": rmse, "mape": mape}


def save_predictions(predictions_path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    output.to_csv(predictions_path, index=False)


def save_json(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def resolve_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "predictions": Path(args.predictions_path) if args.predictions_path else output_dir / "test_predictions.csv",
        "metrics": Path(args.metrics_path) if args.metrics_path else output_dir / "metrics.json",
        "run_config": Path(args.run_config_path) if args.run_config_path else output_dir / "run_config.json",
        "checkpoint": Path(args.checkpoint_path) if args.checkpoint_path else output_dir / "best_model.pt",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def build_run_config(
    args: argparse.Namespace,
    feature_columns: list[str],
    date_column: str,
    has_validation: bool,
) -> dict:
    return {
        "model": args.model,
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "test_dir": args.test_dir,
        "feature_columns": feature_columns,
        "window_size": WINDOW_SIZE,
        "dilations": list(DILATIONS),
        "kernel_size": KERNEL_SIZE,
        "hidden_channels": HIDDEN_CHANNELS,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_rows_per_trip": args.max_rows_per_trip,
        "log_every_batches": args.log_every_batches,
        "target_column": args.target_column,
        "date_column": date_column,
        "has_validation_dir": has_validation,
        "plain_tcn_assumption": (
            "The plain tcn variant uses simpler non-residual layers, while "
            "tcn_r1/tcn_r2/tcn_r3/setcn use the deeper residual block structure."
        ),
    }


def json_safe_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    safe_metrics: dict[str, float | None] = {}
    for key, value in metrics.items():
        safe_metrics[key] = None if np.isnan(value) else value
    return safe_metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_paths = resolve_output_paths(args)
    device = resolve_device(args.device)

    print("Using directory mode.")

    print("Loading train trips...")
    train_trips, date_column, feature_columns = load_trips_from_directory(
        directory_path=args.train_dir,
        target_column=args.target_column,
        date_column=args.date_column,
        feature_columns=args.feature_columns,
        max_rows_per_trip=args.max_rows_per_trip,
    )
    print(f"Loaded {len(train_trips)} train trip CSVs.")

    val_trips: list[tuple[Path, pd.DataFrame]] | None = None
    if args.val_dir is not None and Path(args.val_dir).exists():
        print("Loading validation trips...")
        val_trips, val_date_column, val_features = load_trips_from_directory(
            directory_path=args.val_dir,
            target_column=args.target_column,
            date_column=date_column,
            feature_columns=feature_columns,
            max_rows_per_trip=args.max_rows_per_trip,
        )
        if val_date_column != date_column:
            raise ValueError("Validation directory resolved a different date column.")
        if val_features != feature_columns:
            raise ValueError("Validation directory resolved different feature columns.")
        print(f"Loaded {len(val_trips)} validation trip CSVs.")

    print("Loading test trips...")
    test_trips, test_date_column, test_features = load_trips_from_directory(
        directory_path=args.test_dir,
        target_column=args.target_column,
        date_column=date_column,
        feature_columns=feature_columns,
        max_rows_per_trip=args.max_rows_per_trip,
    )
    if test_date_column != date_column:
        raise ValueError("Test directory resolved a different date column.")
    if test_features != feature_columns:
        raise ValueError("Test directory resolved different feature columns.")
    print(f"Loaded {len(test_trips)} test trip CSVs.")

    print(f"Validation enabled: {val_trips is not None}")
    print(f"Train trip count: {len(train_trips)}")
    print(f"Val trip count: {len(val_trips) if val_trips is not None else 0}")
    print(f"Test trip count: {len(test_trips)}")
    print(f"Train raw rows: {sum(len(trip_df) for _, trip_df in train_trips)}")
    print(f"Val raw rows: {sum(len(trip_df) for _, trip_df in val_trips) if val_trips is not None else 0}")
    print(f"Test raw rows: {sum(len(trip_df) for _, trip_df in test_trips)}")
    if args.max_rows_per_trip is not None:
        print(f"max_rows_per_trip: {args.max_rows_per_trip}")
    print("Selected feature columns:")
    for column in feature_columns:
        print(f"  - {column}")

    scaler_frames = [trip_df for _, trip_df in train_trips]
    scaler_train_df = pd.concat(scaler_frames, ignore_index=True)
    scaler = fit_feature_scaler(scaler_train_df, feature_columns)

    scaled_train_trips = [
        (trip_path, scaler.transform(trip_df, feature_columns))
        for trip_path, trip_df in train_trips
    ]
    scaled_val_trips = (
        [
            (trip_path, scaler.transform(trip_df, feature_columns))
            for trip_path, trip_df in val_trips
        ]
        if val_trips is not None
        else None
    )
    scaled_test_trips = [
        (trip_path, scaler.transform(trip_df, feature_columns))
        for trip_path, trip_df in test_trips
    ]

    train_data = create_sliding_windows_from_trips(
        trips=scaled_train_trips,
        feature_columns=feature_columns,
        target_column=args.target_column,
        split_name="train",
    )
    val_data = (
        create_sliding_windows_from_trips(
            trips=scaled_val_trips,
            feature_columns=feature_columns,
            target_column=args.target_column,
            split_name="val",
        )
        if scaled_val_trips is not None
        else None
    )
    test_data = create_sliding_windows_from_trips(
        trips=scaled_test_trips,
        feature_columns=feature_columns,
        target_column=args.target_column,
        split_name="test",
    )

    print(f"Final train window count: {len(train_data.targets)}")
    print(f"Final val window count: {len(val_data.targets) if val_data is not None else 0}")
    print(f"Final test window count: {len(test_data.targets)}")

    train_loader = build_dataloader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = build_dataloader(test_data, batch_size=args.batch_size, shuffle=False)
    val_loader = (
        build_dataloader(val_data, batch_size=args.batch_size, shuffle=False)
        if val_data is not None
        else None
    )

    model = build_model(
        args.model,
        input_channels=len(feature_columns),
        dropout=args.dropout,
    ).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Using date column: {date_column}")

    best_val_mae = None
    checkpoint_saved = False

    for epoch in range(1, args.epochs + 1):
        train_mae = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            log_every=args.log_every_batches,
        )

        if val_loader is not None:
            val_mae = evaluate_mae(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch}/{args.epochs} - "
                f"train_mae: {train_mae:.6f} - val_mae: {val_mae:.6f}"
            )
            if best_val_mae is None or val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), output_paths["checkpoint"])
                checkpoint_saved = True
        else:
            print(f"Epoch {epoch}/{args.epochs} - train_mae: {train_mae:.6f}")

    if checkpoint_saved:
        model.load_state_dict(torch.load(output_paths["checkpoint"], map_location=device))
        print(f"Loaded best validation checkpoint from {output_paths['checkpoint']}")

    y_true, y_pred = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    save_predictions(output_paths["predictions"], y_true, y_pred)
    save_json(output_paths["metrics"], json_safe_metrics(metrics))
    save_json(
        output_paths["run_config"],
        build_run_config(
            args=args,
            feature_columns=feature_columns,
            date_column=date_column,
            has_validation=val_loader is not None,
        ),
    )

    print(f"Test MAE: {metrics['mae']:.6f}")
    print(f"Test RMSE: {metrics['rmse']:.6f}")
    print(f"Test MAPE: {metrics['mape']:.6f}")
    print(f"Saved test predictions to {output_paths['predictions']}")
    print(f"Saved metrics to {output_paths['metrics']}")
    print(f"Saved run config to {output_paths['run_config']}")
    if checkpoint_saved:
        print(f"Saved best checkpoint to {output_paths['checkpoint']}")


if __name__ == "__main__":
    main()
