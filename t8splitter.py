from pathlib import Path
import pandas as pd

INPUT_ROOT = Path("sampdata_sampled")
OUTPUT_ROOT = Path("sampdata_sampled_top8")

DATE_COLUMN = "time_unix"
TARGET_COLUMN = "electric_powerDemand"

TOP8_FEATURES = [
    "traction_tractionForce",
    "odometry_vehicleSpeed",
    "odometry_wheelSpeed_fr",
    "odometry_wheelSpeed_ml",
    "odometry_wheelSpeed_mr",
    "odometry_wheelSpeed_rr",
    "odometry_wheelSpeed_rl",
    "traction_brakePressure",
]

KEEP_COLUMNS = [DATE_COLUMN] + TOP8_FEATURES + [TARGET_COLUMN]

def process_split(split: str) -> None:
    in_dir = INPUT_ROOT / split
    out_dir = OUTPUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {in_dir}")

    print(f"{split}: {len(files)} files")

    for i, f in enumerate(files, 1):
        df = pd.read_csv(f)

        missing = [col for col in KEEP_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"{f.name} is missing required columns: {missing}")

        reduced_df = df[KEEP_COLUMNS].copy()
        reduced_df.to_csv(out_dir / f.name, index=False)

        if i % 10 == 0 or i == len(files):
            print(f"{split}: processed {i}/{len(files)}")

def main() -> None:
    print("Keeping columns:")
    for col in KEEP_COLUMNS:
        print(f"  - {col}")

    for split in ["train", "val", "test"]:
        process_split(split)

    print(f"Done. Wrote top-8 dataset to {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
