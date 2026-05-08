from pathlib import Path
import pandas as pd

INPUT_ROOT = Path("sampdata_sampled")
OUTPUT_ROOT = Path("sampdata_sampled_5xs")
STEP = 5

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
        sparse_df = df.iloc[::STEP].copy()
        sparse_df.to_csv(out_dir / f.name, index=False)

        if i % 10 == 0 or i == len(files):
            print(f"{split}: processed {i}/{len(files)}")

def main() -> None:
    for split in ["train", "val", "test"]:
        process_split(split)
    print(f"Done. Wrote sparse dataset to {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
