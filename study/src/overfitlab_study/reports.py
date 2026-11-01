"""Tables and figures for the memo: CSV with six significant digits plus Markdown."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SERIES_1 = "#2a78d6"


def format_cell(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return "" if value is None else str(value)


def write_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write ``path`` as CSV and a sibling ``.md`` rendered table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    cells = [[format_cell(row[column]) for column in columns] for row in rows]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(cells)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
        *("| " + " | ".join(row) + " |" for row in cells),
    ]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
