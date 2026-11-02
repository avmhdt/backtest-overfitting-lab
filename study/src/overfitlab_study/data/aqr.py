"""AQR factor workbooks, downloaded by the owner, to the factor_returns schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import openpyxl
import polars as pl

from overfitlab_study.data import french, tables
from overfitlab_study.data.manifest import write_manifest

SOURCE = "aqr"


@dataclass(frozen=True)
class Workbook:
    dataset: str
    file: str
    sheet: str
    frequency: str
    columns: tuple[str, ...] | None


def workbooks_from_config(items: Sequence[Mapping[str, Any]]) -> list[Workbook]:
    return [
        Workbook(
            dataset=item["dataset"],
            file=item["file"],
            sheet=item["sheet"],
            frequency=item.get("frequency", "M"),
            columns=tuple(item["columns"]) if item.get("columns") else None,
        )
        for item in items
    ]


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%m/%d/%Y").date()
        except ValueError:
            return None
    return None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def parse_sheet(path: Path, workbook: Workbook) -> pl.DataFrame:
    """Long factor_returns from one sheet: the header is the row above the first dated row."""
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        header: Sequence[object] = ()
        names: dict[int, str] = {}
        records: list[dict[str, object]] = []
        for row in book[workbook.sheet].iter_rows(values_only=True):
            day = _as_date(row[0]) if row else None
            if day is None:
                if records:
                    break
                if sum(cell is not None for cell in row) >= 2:
                    header = row
                continue
            if not names:
                names = {
                    i: str(name).strip() for i, name in enumerate(header) if i and name is not None
                }
                if workbook.columns is not None:
                    names = {i: name for i, name in names.items() if name in workbook.columns}
                if not names:
                    raise ValueError(f"{path.name}: no columns found in {workbook.sheet}")
            stamp = tables.month_end(day) if workbook.frequency == "M" else day
            for i, name in names.items():
                records.append(
                    {
                        "date": stamp,
                        "dataset": workbook.dataset,
                        "factor": name,
                        "region": None,
                        "frequency": workbook.frequency,
                        "ret": _as_float(row[i]),
                    }
                )
    finally:
        book.close()
    return pl.DataFrame(records, schema=french.FACTOR_RETURNS_SCHEMA)


def ingest(
    workbooks: Sequence[Workbook],
    *,
    aqr_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
) -> dict[str, int]:
    rows: dict[str, int] = {}
    files: list[Path] = []
    for workbook in workbooks:
        path = aqr_dir / workbook.file
        frame = parse_sheet(path, workbook)
        tables.write_table(frame, tables.factor_path(processed_dir, SOURCE, workbook.dataset))
        for (factor,), part in frame.group_by(["factor"], maintain_order=True):
            flags = tables.extreme_returns(
                f"{workbook.dataset}/{factor}",
                part.get_column("date").to_list(),
                part.get_column("ret").to_list(),
                tables.MONTHLY_FLAG,
            )
            tables.append_flags(interim_dir, SOURCE, flags)
        rows[workbook.dataset] = frame.height
        files.append(path)
    write_manifest(
        aqr_dir,
        source=SOURCE,
        files=files,
        row_counts=rows,
        units="decimal monthly returns as published",
        origin="manual download from aqr.com/Insights/Datasets",
    )
    return rows
