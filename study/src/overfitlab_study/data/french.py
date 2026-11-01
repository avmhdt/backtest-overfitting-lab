"""Ken French Data Library: download and parse to the factor_returns schema."""

from __future__ import annotations

import calendar
import io
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import polars as pl

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
SMOKE_DATASET = "F-F_Research_Data_Factors"
SENTINELS = (-99.99, -999.0)

FACTOR_RETURNS_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "date": pl.Date,
    "dataset": pl.String,
    "factor": pl.String,
    "region": pl.String,
    "frequency": pl.String,
    "ret": pl.Float64,
}

_ROW = re.compile(r"^\s*(\d{4}|\d{6}|\d{8})\s*,")


@dataclass(frozen=True)
class FrenchTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, tuple[float | None, ...]], ...]


def dataset_url(dataset: str) -> str:
    return f"{BASE_URL}/{dataset}_CSV.zip"


def download(dataset: str, raw_dir: Path, client: httpx.Client) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    response = client.get(dataset_url(dataset), follow_redirects=True, timeout=60.0)
    response.raise_for_status()
    path = raw_dir / f"{dataset}_CSV.zip"
    path.write_bytes(response.content)
    return path


def read_zip_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        return io.TextIOWrapper(archive.open(name), encoding="latin-1").read()


def _parse_value(text: str) -> float | None:
    """Percent to decimal (DQ-03); sentinels and blanks become null."""
    text = text.strip()
    if not text:
        return None
    value = float(text)
    if any(abs(value - sentinel) < 1e-9 for sentinel in SENTINELS):
        return None
    return value / 100.0


def parse_tables(text: str) -> list[FrenchTable]:
    """Split a French CSV into tables: a text title, a header row, then dated rows."""
    tables: list[FrenchTable] = []
    title = ""
    columns: tuple[str, ...] | None = None
    rows: list[tuple[str, tuple[float | None, ...]]] = []

    def flush() -> None:
        nonlocal columns, rows
        if columns is not None and rows:
            tables.append(FrenchTable(title, columns, tuple(rows)))
        columns, rows = None, []

    for line in text.splitlines():
        if _ROW.match(line) and columns is not None:
            stamp, *values = [part.strip() for part in line.split(",")]
            rows.append((stamp, tuple(_parse_value(value) for value in values)))
        elif line.strip().startswith(","):
            flush()
            columns = tuple(part.strip() for part in line.split(",")[1:])
        elif line.strip():
            flush()
            title = line.strip()
        else:
            flush()
    flush()
    return tables


def _to_date(stamp: str) -> tuple[date, str] | None:
    """Daily rows keep their date; monthly rows are stamped at month end (DQ-04)."""
    if len(stamp) == 8:
        return date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:])), "D"
    if len(stamp) == 6:
        year, month = int(stamp[:4]), int(stamp[4:])
        return date(year, month, calendar.monthrange(year, month)[1]), "M"
    return None


def to_factor_returns(dataset: str, text: str) -> pl.DataFrame:
    """Long factor_returns table. Annual tables are dropped; the first table wins."""
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for table in parse_tables(text):
        first = _to_date(table.rows[0][0])
        if first is None or first[1] in seen:
            continue
        seen.add(first[1])
        for stamp, values in table.rows:
            parsed = _to_date(stamp)
            if parsed is None:
                raise ValueError(f"mixed date formats in {dataset}: {stamp}")
            day, frequency = parsed
            for factor, value in zip(table.columns, values, strict=True):
                records.append(
                    {
                        "date": day,
                        "dataset": dataset,
                        "factor": factor,
                        "region": None,
                        "frequency": frequency,
                        "ret": value,
                    }
                )
    return pl.DataFrame(records, schema=FACTOR_RETURNS_SCHEMA)
