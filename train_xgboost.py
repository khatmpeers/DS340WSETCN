from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

from model import WINDOW_SIZE


TARGET_COLUMN = "electric_powerDemand"
TRAIN_YEARS = (2019, 2020, 2021)
TEST_YEARS = (2022,)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost baseline for DTB energy consumption prediction."
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to a single CSV file or a directory containing one CSV per trip.",
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
    parser.add_argument("--n_estimators", type=int, default=300, help="Number of boosting rounds.")
    parser.add_argument("--max_depth", type=int, default=6, help="Maximum tree depth.")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Boosting learning rate.")
    parser.add_argument("--subsample", type=float, default=0.8, help="Row subsampling ratio.")
    parser.add_argument(
        "--colsample_bytree",
        type=float,
        default=0.8,
        help="Feature subsampling ratio per tree.",
    )
    parser.add_argument(
        "--reg_alpha",
        type=float,
        default=0.0,
        help="L1 regularization term on weights.",
    )
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=1.0,
        help="L2 regularization term on weights.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel threads for XGBoost.",
    )
    parser.add_argument(
        "--max_train_trips",
        type=int,
        default=None,
        help="Optional cap on the number of training trips to use after year splitting.",
    )
    parser.add_argument(
        "--max_test_trips",
        type=int,
        default=None,
        help="Optional cap on the number of testing trips to use after year splitting.",
    )
    parser.add_argument(
        "--max_rows_per_trip",
        type=int,
        default=None,
        help="Optional cap on the number of rows to keep from each trip after sorting by time.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs_xgboost",
        help="Directory for predictions, metrics, SHAP outputs, and run config.",
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
        "--lag_shap_path",
        default=None,
        help="Optional override for the lag-level SHAP CSV path.",
    )
    parser.add_argument(
        "--feature_shap_path",
        default=None,
        help="Optional override for the aggregated feature-level SHAP CSV path.",
    )
    parser.add_argument(
        "--run_config_path",
        default=None,
        help="Optional override for the run config JSON path.",
    )
    parser.add_argument(
        "--shap_plot_path",
        default=None,
        help="Optional override for the SHAP summary plot path.",
    )
    parser.add_argument(
        "--shap_sample_size",
        type=int,
        default=None,
        help="Optional number of test samples to use for SHAP computation.",
    )
    return parser.parse_args()


def infer_date_column(dataframe: pd.DataFrame) -> str:
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


def load_trip_dataframes(
    data_path: str,
    target_column: str,
    date_column: str | None,
    feature_columns: list[str] | None,
    max_rows_per_trip: int | None,
    max_train_trips: int = 1000,
) -> tuple[list[tuple[Path, pd.DataFrame]], str, list[str]]:
    source_path = Path(data_path)
    if not source_path.exists():
        raise ValueError(f"Data path does not exist: {data_path}")

    csv_paths = [source_path] if source_path.is_file() else sorted(source_path.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"No CSV files found in {data_path}")

    trip_entries: list[tuple[Path, pd.DataFrame, pd.Timestamp]] = []
    resolved_date_column: str | None = None
    selected_features: list[str] | None = None

    i = 0

    for csv_path in csv_paths:
        if i >= max_train_trips:
            break
        trip_df, trip_date_column, trip_features = load_trip_dataframe(
            csv_path=csv_path,
            target_column=target_column,
            date_column=date_column,
            feature_columns=feature_columns,
            max_rows_per_trip=max_rows_per_trip,
        )
        if trip_df.empty:
            continue

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

        trip_entries.append((csv_path, trip_df, trip_df[trip_date_column].iloc[0]))
        i += 1 

    if not trip_entries:
        raise ValueError("No non-empty trip CSV files with valid timestamps were found.")

    trip_entries.sort(key=lambda entry: entry[2])
    trips = [(csv_path, trip_df) for csv_path, trip_df, _ in trip_entries]
    return trips, resolved_date_column or "", selected_features or []


def split_trips_by_year(
    trips: list[tuple[Path, pd.DataFrame]],
    date_column: str,
) -> tuple[list[tuple[Path, pd.DataFrame]], list[tuple[Path, pd.DataFrame]]]:
    train_trips: list[tuple[Path, pd.DataFrame]] = []
    test_trips: list[tuple[Path, pd.DataFrame]] = []

    for trip_path, trip_df in trips:
        trip_years = set(trip_df[date_column].dt.year.unique().tolist())
        if trip_years.issubset(set(TRAIN_YEARS)):
            train_trips.append((trip_path, trip_df))
        elif trip_years.issubset(set(TEST_YEARS)):
            test_trips.append((trip_path, trip_df))

    if not train_trips:
        raise ValueError("No trip files found entirely within training years 2019, 2020, and 2021.")
    if not test_trips:
        raise ValueError("No trip files found entirely within testing year 2022.")

    return train_trips, test_trips


def build_flattened_windows_for_trip(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    window_size: int = WINDOW_SIZE,
) -> tuple[list[np.ndarray], list[float]]:
    features = dataframe[feature_columns].to_numpy(dtype=np.float32)
    targets = dataframe[target_column].to_numpy(dtype=np.float32)

    if len(dataframe) <= window_size:
        return [], []

    flattened_rows: list[np.ndarray] = []
    window_targets: list[float] = []

    for start_index in range(len(dataframe) - window_size):
        end_index = start_index + window_size
        flattened_rows.append(features[start_index:end_index].reshape(-1))
        window_targets.append(float(targets[end_index]))

    return flattened_rows, window_targets


def build_flattened_windows_from_trips(
    trips: list[tuple[Path, pd.DataFrame]],
    feature_columns: list[str],
    target_column: str,
    window_size: int = WINDOW_SIZE,
) -> tuple[pd.DataFrame, np.ndarray]:
    all_rows: list[np.ndarray] = []
    all_targets: list[float] = []
    total_trips = len(trips)

    for trip_index, (_, trip_df) in enumerate(trips, start=1):
        trip_rows, trip_targets = build_flattened_windows_for_trip(
            dataframe=trip_df,
            feature_columns=feature_columns,
            target_column=target_column,
            window_size=window_size,
        )
        all_rows.extend(trip_rows)
        all_targets.extend(trip_targets)

        if trip_index % 10 == 0 or trip_index == total_trips:
            print(
                f"Processed trips {trip_index}/{total_trips} - "
                f"cumulative windows: {len(all_rows)}"
            )

    if not all_rows:
        raise ValueError(
            f"No eligible sliding windows were produced with window_size={window_size}."
        )

    column_names = make_lagged_column_names(feature_columns, window_size)
    x = pd.DataFrame(all_rows, columns=column_names)
    y = np.asarray(all_targets, dtype=np.float32)
    return x, y


def make_lagged_column_names(feature_columns: list[str], window_size: int) -> list[str]:
    column_names = []
    for lag_index in range(window_size):
        lag_suffix = window_size - 1 - lag_index
        for feature_name in feature_columns:
            column_names.append(f"{feature_name}_t-{lag_suffix}")
    return column_names


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    nonzero_mask = y_true != 0
    if np.any(nonzero_mask):
        mape: float | None = float(
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]))
            * 100.0
        )
    else:
        mape = None

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
        "lag_shap": Path(args.lag_shap_path) if args.lag_shap_path else output_dir / "shap_lag_importance.csv",
        "feature_shap": Path(args.feature_shap_path) if args.feature_shap_path else output_dir / "shap_feature_importance.csv",
        "run_config": Path(args.run_config_path) if args.run_config_path else output_dir / "run_config.json",
        "shap_plot": Path(args.shap_plot_path) if args.shap_plot_path else output_dir / "shap_summary.png",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def build_run_config(
    args: argparse.Namespace,
    feature_columns: list[str],
    date_column: str,
) -> dict:
    return {
        "model": "xgboost",
        "feature_columns": feature_columns,
        "window_size": WINDOW_SIZE,
        "target_column": args.target_column,
        "date_column": date_column,
        "train_years": list(TRAIN_YEARS),
        "test_years": list(TEST_YEARS),
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
    }


def compute_shap_values(
    model: XGBRegressor,
    x_test: pd.DataFrame,
) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    return np.asarray(shap_values, dtype=np.float32)


def save_lag_level_importance(
    lag_shap_path: Path,
    x_test: pd.DataFrame,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    lag_importance = pd.DataFrame(
        {
            "feature_lag": x_test.columns,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    lag_importance.to_csv(lag_shap_path, index=False)
    return lag_importance


def save_feature_level_importance(
    feature_shap_path: Path,
    lag_importance: pd.DataFrame,
) -> pd.DataFrame:
    aggregated = lag_importance.copy()
    aggregated["feature"] = aggregated["feature_lag"].str.replace(r"_t-\d+$", "", regex=True)
    feature_importance = (
        aggregated.groupby("feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    feature_importance.to_csv(feature_shap_path, index=False)
    return feature_importance


def try_save_shap_summary_plot(
    shap_plot_path: Path,
    shap_values: np.ndarray,
    x_test: pd.DataFrame,
) -> bool:
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        shap.summary_plot(shap_values, x_test, show=False)
        plt.tight_layout()
        plt.savefig(shap_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    output_paths = resolve_output_paths(args)

    print("Loading trip CSVs...")
    trips, date_column, feature_columns = load_trip_dataframes(
        data_path=args.data_path,
        target_column=args.target_column,
        date_column=args.date_column,
        feature_columns=args.feature_columns,
        max_rows_per_trip=args.max_rows_per_trip,
    )
    print("Finished loading trip CSVs.")
    if args.max_rows_per_trip is not None:
        print(f"max_rows_per_trip: {args.max_rows_per_trip}")
    print("Splitting trips by year...")
    train_trips, test_trips = split_trips_by_year(trips, date_column)
    if args.max_train_trips is not None:
        train_trips = train_trips[: args.max_train_trips]
    if args.max_test_trips is not None:
        test_trips = test_trips[: args.max_test_trips]

    print(f"Training trips selected: {len(train_trips)}")
    print(f"Testing trips selected: {len(test_trips)}")
    print(f"Total raw training rows: {sum(len(trip_df) for _, trip_df in train_trips)}")
    print(f"Total raw testing rows: {sum(len(trip_df) for _, trip_df in test_trips)}")
    print("Building training windows...")
    x_train, y_train = build_flattened_windows_from_trips(
        train_trips,
        feature_columns,
        args.target_column,
    )
    print("Building testing windows...")
    x_test, y_test = build_flattened_windows_from_trips(
        test_trips,
        feature_columns,
        args.target_column,
    )

    print(f"Using date column: {date_column}")
    print("Using feature columns:")
    for column in feature_columns:
        print(f"  - {column}")
    print(f"Training trips used: {len(train_trips)}")
    print(f"Testing trips used: {len(test_trips)}")
    print(f"Training rows after windowing: {len(x_train)}")
    print(f"Testing rows after windowing: {len(x_test)}")
    print(f"Total test rows: {len(x_test)}")

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = compute_metrics(y_test, y_pred)
    save_predictions(output_paths["predictions"], y_test, y_pred)
    save_json(output_paths["metrics"], metrics)
    save_json(output_paths["run_config"], build_run_config(args, feature_columns, date_column))

    if args.shap_sample_size is not None and len(x_test) > args.shap_sample_size:
        x_test_shap = x_test.sample(n=args.shap_sample_size, random_state=42)
    else:
        x_test_shap = x_test

    print(f"SHAP sample size used: {len(x_test_shap)}")

    shap_values = compute_shap_values(model, x_test_shap)
    lag_importance = save_lag_level_importance(output_paths["lag_shap"], x_test_shap, shap_values)
    save_feature_level_importance(output_paths["feature_shap"], lag_importance)
    plot_saved = try_save_shap_summary_plot(output_paths["shap_plot"], shap_values, x_test_shap)

    print(f"Test MAE: {metrics['mae']:.6f}")
    print(f"Test RMSE: {metrics['rmse']:.6f}")
    if metrics["mape"] is None:
        print("Test MAPE: undefined because all target values are zero.")
    else:
        print(f"Test MAPE: {metrics['mape']:.6f}")
    print(f"Saved predictions to {output_paths['predictions']}")
    print(f"Saved metrics to {output_paths['metrics']}")
    print(f"Saved lag-level SHAP importances to {output_paths['lag_shap']}")
    print(f"Saved aggregated feature-level SHAP importances to {output_paths['feature_shap']}")
    print(f"Saved run config to {output_paths['run_config']}")
    if plot_saved:
        print(f"Saved SHAP summary plot to {output_paths['shap_plot']}")
    else:
        print("Skipped SHAP summary plot export.")


if __name__ == "__main__":
    main()
