#!/usr/bin/env python3
"""Rename files by moving the third underscore-separated token to the end.

Example:
    a_b_X_c_d_e_f.txt -> a_b_c_d_e_f_X.txt

Usage:
    python3 rename_files_move_third_token.py /path/to/folder
    python3 rename_files_move_third_token.py /path/to/folder --dry-run
"""

import argparse
from pathlib import Path


def rename_files(folder: Path, ext: str, dry_run: bool) -> int:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    renamed_count = 0
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() != ext.lower():
            continue

        parts = path.stem.split("_")
        if len(parts) < 4:
            continue

        new_parts = parts[:2] + parts[3:] + [parts[2]]
        new_name = "_".join(new_parts) + path.suffix
        new_path = path.with_name(new_name)

        if new_path == path:
            continue

        if new_path.exists():
            print(f"Skipping existing target: {new_path}")
            continue

        print(f"{path.name} -> {new_path.name}")
        if not dry_run:
            path.rename(new_path)
        renamed_count += 1

    return renamed_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename files by moving the third underscore-separated token to the end of the name."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing files to rename.",
    )
    parser.add_argument(
        "--ext",
        default=".txt",
        help="File extension to process (default: .txt).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show proposed renames without applying them.",
    )

    args = parser.parse_args()
    count = rename_files(args.folder, args.ext, args.dry_run)
    print(f"Renamed {count} file(s){' (dry run)' if args.dry_run else ''}.")


if __name__ == "__main__":
    main()
