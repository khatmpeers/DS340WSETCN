import pandas as pd
from pathlib import Path

INPUT_ROOT = Path("sampdata_sampled")
OUTPUT_ROOT = Path("sampdata_sampled_2xs")

STEP = 2  # change to 5 if you want 5x sparsity

def process_split(split):
    in_dir = INPUT_ROOT / split
    out_dir = OUTPUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(in_dir.glob("*.csv"))
    print(f"{split}: {len(files)} files")

    for i, f in enumerate(files, 1):
        df = pd.read_csv(f)

        # downsample
        df_sparse = df.iloc[::STEP].copy()

        out_path = out_dir / f.name
        df_sparse.to_csv(out_path, index=False)

        if i % 10 == 0:
            print(f"{split}: processed {i}/{len(files)}")

def main():
    for split in ["train", "val", "test"]:
        process_split(split)

    print("Done.")

if __name__ == "__main__":
    main()
