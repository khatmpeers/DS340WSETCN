from __future__ import annotations

import argparse
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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
    bus: str
    start_year: int
    start_month: int
    year_month: str
    start_stamp: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a representative ZTBus subset by bus and year."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing mission CSV files.",
    )
    parser.add_argument(
        "--bus",
        default="B183",
        help="Bus to keep, e.g. B183 or B208. Default: B183",
    )
    parser.add_argument(
        "--train_n",
        type=int,
        default=300,
        help="Number of train missions to sample from 2019-2021.",
    )
    parser.add_argument(
        "--val_n",
        type=int,
        default=60,
        help="Number of validation missions to sample from 2019-2021.",
    )
    parser.add_argument(
        "--test_n",
        type=int,
        default=75,
        help="Number of test missions to sample from 2022.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output_dir",
        default="sampled_subset",
        help="Directory to write split manifests.",
    )
    return parser.parse_args()


def parse_mission_file(path: Path) -> MissionFile | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None

    bus = f"B{match.group('bus')}"
    start_date = match.group("start_date")
    start_time = match.group("start_time").replace("-", ":")
    start_year = int(start_date[:4])
    start_month = int(start_date[5:7])
    year_month = start_date[:7]
    start_stamp = f"{start_date} {start_time}"

    return MissionFile(
        path=path,
        bus=bus,
        start_year=start_year,
        start_month=start_month,
        year_month=year_month,
        start_stamp=start_stamp,
    )


def load_missions(data_dir: Path, bus: str) -> list[MissionFile]:
    missions: list[MissionFile] = []
    for path in sorted(data_dir.glob("*.csv")):
        parsed = parse_mission_file(path)
        if parsed is None:
            continue
        if parsed.bus != bus:
            continue
        missions.append(parsed)

    if not missions:
        raise ValueError(f"No mission CSVs found for bus {bus} in {data_dir}")
    return missions


def split_pools(
    missions: Iterable[MissionFile],
) -> tuple[list[MissionFile], list[MissionFile]]:
    train_pool = [m for m in missions if m.start_year in {2019, 2020, 2021}]
    test_pool = [m for m in missions if m.start_year == 2022]

    if not train_pool:
        raise ValueError("No train-pool missions found for years 2019-2021.")
    if not test_pool:
        raise ValueError("No test-pool missions found for year 2022.")
    return train_pool, test_pool


def stratified_sample_by_month(
    missions: list[MissionFile],
    n: int,
    rng: random.Random,
) -> list[MissionFile]:
    """
    Month-aware sampling:
    - group by YYYY-MM
    - allocate sample count roughly proportional to group size
    - ensure groups with available missions can contribute
    - fill leftover slots globally
    """
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

    # Fill remaining slots from months that still have unused missions
    remaining_needed = n - len(selected)
    if remaining_needed > 0:
        leftovers: list[MissionFile] = []
        for month, items in groups.items():
            leftovers.extend(items[allocations[month]:])
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining_needed])

    # If proportional rounding overshot somehow, trim randomly but reproducibly
    if len(selected) > n:
        rng.shuffle(selected)
        selected = selected[:n]

    return sorted(selected, key=lambda m: m.start_stamp)


def remove_selected(
    pool: list[MissionFile],
    selected: list[MissionFile],
) -> list[MissionFile]:
    selected_paths = {m.path for m in selected}
    return [m for m in pool if m.path not in selected_paths]


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

    print(f"\n{name}: {len(missions)} missions")
    for month in sorted(by_month):
        print(f"  {month}: {by_month[month]}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    missions = load_missions(data_dir, args.bus)
    train_pool, test_pool = split_pools(missions)

    # Validation comes from the train years only
    val_split = stratified_sample_by_month(train_pool, args.val_n, rng)
    remaining_train_pool = remove_selected(train_pool, val_split)
    train_split = stratified_sample_by_month(remaining_train_pool, args.train_n, rng)
    test_split = stratified_sample_by_month(test_pool, args.test_n, rng)

    write_manifest(output_dir / "train_files.txt", train_split)
    write_manifest(output_dir / "val_files.txt", val_split)
    write_manifest(output_dir / "test_files.txt", test_split)

    print(f"Bus filter: {args.bus}")
    print(f"Total eligible missions for {args.bus}: {len(missions)}")
    print(f"Train pool (2019-2021): {len(train_pool)}")
    print(f"Test pool (2022): {len(test_pool)}")

    summarize("TRAIN", train_split)
    summarize("VAL", val_split)
    summarize("TEST", test_split)

    print("\nWrote:")
    print(f"  {output_dir / 'train_files.txt'}")
    print(f"  {output_dir / 'val_files.txt'}")
    print(f"  {output_dir / 'test_files.txt'}")


if __name__ == "__main__":
    main()
