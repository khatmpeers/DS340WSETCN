from __future__ import annotations

import argparse
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path


FILENAME_RE = re.compile(
    r"^(B(?P<bus>\d+))_"
    r"(?P<start_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<start_time>\d{2}-\d{2}-\d{2})_"
    r"(?P<end_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<end_time>\d{2}-\d{2}-\d{2})\.csv$"
)


@dataclass(frozen=True)
class MissionFile:
    path: Path
    year_month: str
    start_stamp: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample smaller month-aware subsets from existing train/val/test partitions."
    )
    parser.add_argument(
        "--partition_dir",
        required=True,
        help="Directory containing train/, val/, and test/ subdirectories.",
    )
    parser.add_argument(
        "--train_n",
        type=int,
        default=100,
        help="Number of train CSVs to sample.",
    )
    parser.add_argument(
        "--val_n",
        type=int,
        default=25,
        help="Number of validation CSVs to sample.",
    )
    parser.add_argument(
        "--test_n",
        type=int,
        default=40,
        help="Number of test CSVs to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output_dir",
        default="subset_manifests_small",
        help="Directory to write sampled manifest files.",
    )
    return parser.parse_args()


def parse_mission_file(path: Path) -> MissionFile | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None

    start_date = match.group("start_date")
    start_time = match.group("start_time").replace("-", ":")
    year_month = start_date[:7]
    start_stamp = f"{start_date} {start_time}"

    return MissionFile(
        path=path,
        year_month=year_month,
        start_stamp=start_stamp,
    )


def load_partition(partition_path: Path) -> list[MissionFile]:
    if not partition_path.exists():
        raise ValueError(f"Partition directory not found: {partition_path}")

    missions: list[MissionFile] = []
    for csv_path in sorted(partition_path.glob("*.csv")):
        parsed = parse_mission_file(csv_path)
        if parsed is not None:
            missions.append(parsed)

    if not missions:
        raise ValueError(f"No valid mission CSVs found in {partition_path}")

    return missions


def stratified_sample_by_month(
    missions: list[MissionFile],
    n: int,
    rng: random.Random,
) -> list[MissionFile]:
    if n <= 0:
        return []
    if n >= len(missions):
        return sorted(missions, key=lambda m: m.start_stamp)

    groups: dict[str, list[MissionFile]] = {}
    for mission in missions:
        groups.setdefault(mission.year_month, []).append(mission)

    for month in groups:
        rng.shuffle(groups[month])

    total = len(missions)
    allocations: dict[str, int] = {}

    # Initial proportional allocation
    for month, items in groups.items():
        allocations[month] = min(len(items), math.floor(n * len(items) / total))

    selected: list[MissionFile] = []
    for month, count in allocations.items():
        selected.extend(groups[month][:count])

    remaining_needed = n - len(selected)
    if remaining_needed > 0:
        leftovers: list[MissionFile] = []
        for month, items in groups.items():
            leftovers.extend(items[allocations[month]:])
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining_needed])

    if len(selected) > n:
        rng.shuffle(selected)
        selected = selected[:n]

    return sorted(selected, key=lambda m: m.start_stamp)


def write_manifest(path: Path, missions: list[MissionFile]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for mission in missions:
            f.write(str(mission.path.resolve()))
            f.write("\n")


def summarize(name: str, missions: list[MissionFile]) -> None:
    by_month: dict[str, int] = {}
    for mission in missions:
        by_month[mission.year_month] = by_month.get(mission.year_month, 0) + 1

    print(f"\n{name}: {len(missions)} files")
    for month in sorted(by_month):
        print(f"  {month}: {by_month[month]}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    partition_dir = Path(args.partition_dir)
    output_dir = Path(args.output_dir)

    train_all = load_partition(partition_dir / "train")
    val_all = load_partition(partition_dir / "val")
    test_all = load_partition(partition_dir / "test")

    train_subset = stratified_sample_by_month(train_all, args.train_n, rng)
    val_subset = stratified_sample_by_month(val_all, args.val_n, rng)
    test_subset = stratified_sample_by_month(test_all, args.test_n, rng)

    write_manifest(output_dir / "train_files.txt", train_subset)
    write_manifest(output_dir / "val_files.txt", val_subset)
    write_manifest(output_dir / "test_files.txt", test_subset)

    print("Original partition sizes:")
    print(f"  train: {len(train_all)}")
    print(f"  val:   {len(val_all)}")
    print(f"  test:  {len(test_all)}")

    summarize("TRAIN SUBSET", train_subset)
    summarize("VAL SUBSET", val_subset)
    summarize("TEST SUBSET", test_subset)

    print("\nWrote:")
    print(f"  {output_dir / 'train_files.txt'}")
    print(f"  {output_dir / 'val_files.txt'}")
    print(f"  {output_dir / 'test_files.txt'}")


if __name__ == "__main__":
    main()
