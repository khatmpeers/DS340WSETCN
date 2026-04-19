from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize dataset splits from manifest files."
    )
    parser.add_argument(
        "--manifest_dir",
        required=True,
        help="Directory containing train_files.txt, val_files.txt, test_files.txt",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write train/ val/ test/ folders",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Whether to copy files or create symlinks (default: copy)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if they exist",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[Path]:
    if not path.exists():
        raise ValueError(f"Missing manifest file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [Path(line.strip()) for line in f if line.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def materialize_split(
    file_paths: list[Path],
    dest_dir: Path,
    mode: str,
    overwrite: bool,
) -> None:
    ensure_dir(dest_dir)

    for src in file_paths:
        if not src.exists():
            print(f"WARNING: missing file, skipping: {src}")
            continue

        dest = dest_dir / src.name

        if dest.exists():
            if overwrite:
                if dest.is_file() or dest.is_symlink():
                    dest.unlink()
                else:
                    continue
            else:
                continue

        if mode == "copy":
            shutil.copy2(src, dest)
        elif mode == "symlink":
            dest.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()

    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)

    train_manifest = manifest_dir / "train_files.txt"
    val_manifest = manifest_dir / "val_files.txt"
    test_manifest = manifest_dir / "test_files.txt"

    train_files = read_manifest(train_manifest)
    val_files = read_manifest(val_manifest)
    test_files = read_manifest(test_manifest)

    print("Materializing dataset...")
    print(f"Mode: {args.mode}")
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")

    materialize_split(
        train_files,
        output_dir / "train",
        args.mode,
        args.overwrite,
    )
    materialize_split(
        val_files,
        output_dir / "val",
        args.mode,
        args.overwrite,
    )
    materialize_split(
        test_files,
        output_dir / "test",
        args.mode,
        args.overwrite,
    )

    print("\nDone.")
    print(f"Train dir: {output_dir / 'train'}")
    print(f"Val dir: {output_dir / 'val'}")
    print(f"Test dir: {output_dir / 'test'}")


if __name__ == "__main__":
    main()
