"""Fail if tracked files break the data and size rules (NFR-07)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAX_BYTES = 1_000_000
BLOCKED_SUFFIXES = {".parquet", ".zip", ".xlsx", ".h5"}


def find_problems(paths: list[str], sizes: dict[str, int]) -> list[str]:
    problems: list[str] = []
    for path in paths:
        posix = Path(path).as_posix()
        if posix.startswith("data/"):
            problems.append(f"{posix}: tracked file under data/")
        if Path(posix).suffix.lower() in BLOCKED_SUFFIXES and not posix.startswith(
            "tests/fixtures/"
        ):
            problems.append(f"{posix}: licensed or binary data extension")
        if sizes.get(path, 0) > MAX_BYTES:
            problems.append(f"{posix}: larger than 1 MB")
    return problems


def main() -> int:
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    sizes = {path: Path(path).stat().st_size for path in tracked if Path(path).exists()}
    problems = find_problems(tracked, sizes)
    for problem in problems:
        print(problem)
    print(f"checked {len(tracked)} tracked files, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
