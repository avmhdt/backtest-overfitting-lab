"""Canonical Parquet tables and the data-quality rules shared by the ingest modules."""

from __future__ import annotations

import calendar
import csv
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

PRICES_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "date": pl.Date,
    "symbol": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "adjusted": pl.Boolean,
    "source": pl.String,
}
SIGNAL_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "date": pl.Date,
    "signal": pl.String,
    "portfolio": pl.String,
    "weighting": pl.String,
    "ret": pl.Float64,
    "sample_start": pl.Date,
    "sample_end": pl.Date,
}
DAILY_FLAG = 0.5
MONTHLY_FLAG = 1.0
MAX_GAP_DAYS = 10


def month_end(day: date) -> date:
    """Stamp a monthly observation on the last calendar day of its month (DQ-04)."""
    return date(day.year, day.month, calendar.monthrange(day.year, day.month)[1])


def write_table(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")


def prices_path(processed_dir: Path, source: str, symbol: str) -> Path:
    return (
        processed_dir / "prices_daily" / f"source={source}" / f"symbol={symbol}" / "part-0.parquet"
    )


def factor_path(processed_dir: Path, source: str, dataset: str) -> Path:
    return (
        processed_dir
        / "factor_returns"
        / f"source={source}"
        / f"dataset={dataset}"
        / "part-0.parquet"
    )


def signal_path(processed_dir: Path, release: str) -> Path:
    return processed_dir / "signal_returns" / f"release={release}" / "part-0.parquet"


def append_flags(interim_dir: Path, source: str, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Record flagged observations (DQ-05). They stay in the data."""
    path = interim_dir / source / "flags.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["series", "date", "kind", "value"])
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
    return path


def extreme_returns(
    series: str, dates: list[date], returns: list[float | None], limit: float
) -> list[dict[str, Any]]:
    """Observations beyond the DQ-05 limit."""
    return [
        {"series": series, "date": str(day), "kind": "extreme_return", "value": value}
        for day, value in zip(dates, returns, strict=True)
        if value is not None and abs(value) > limit
    ]


def gaps(series: str, dates: list[date], limit: int = MAX_GAP_DAYS) -> list[dict[str, Any]]:
    """Calendar gaps wider than five trading days (DQ-01)."""
    return [
        {"series": series, "date": str(later), "kind": "gap_days", "value": (later - earlier).days}
        for earlier, later in zip(dates[:-1], dates[1:], strict=True)
        if (later - earlier).days > limit
    ]


def price_flags(frame: pl.DataFrame, series: str) -> list[dict[str, Any]]:
    """Extreme daily returns and gaps of one symbol."""
    dates = frame.get_column("date").to_list()
    returns = frame.get_column("close").pct_change().to_list()
    return [*extreme_returns(series, dates, returns, DAILY_FLAG), *gaps(series, dates)]
