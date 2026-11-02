"""The owner's local Stooq snapshot: index the tree once and convert the universe to Parquet."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from overfitlab_study.data import tables
from overfitlab_study.data.manifest import write_manifest

logger = logging.getLogger(__name__)
SOURCE = "stooq"
RAW_COLUMNS = {
    "<DATE>": "date",
    "<OPEN>": "open",
    "<HIGH>": "high",
    "<LOW>": "low",
    "<CLOSE>": "close",
    "<VOL>": "volume",
}


@dataclass(frozen=True)
class AdjustmentCheck:
    """A close known from the unadjusted tape, for DQ-06."""

    symbol: str
    day: date
    unadjusted_close: float


def build_index(stooq_dir: Path, cache: Path | None = None) -> dict[str, Path]:
    """Map lower-case tickers to files by walking the tree once, never guessing paths."""
    if cache is not None and cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        return {ticker: Path(path) for ticker, path in stored.items()}
    index = {path.stem.lower(): path for path in stooq_dir.rglob("*.txt")}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        payload = {ticker: str(path) for ticker, path in index.items()}
        cache.write_text(json.dumps(payload), encoding="utf-8")
    return index


def parse_file(path: Path, symbol: str, *, adjusted: bool) -> pl.DataFrame:
    """Read a Stooq bar file, driven by its header names."""
    raw = pl.read_csv(path, infer_schema_length=0)
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = raw.select(
        pl.col("<DATE>").str.strptime(pl.Date, "%Y%m%d").alias("date"),
        pl.lit(symbol).alias("symbol"),
        *(
            pl.col(name).cast(pl.Float64).alias(RAW_COLUMNS[name])
            for name in RAW_COLUMNS
            if name != "<DATE>"
        ),
        pl.lit(adjusted).alias("adjusted"),
        pl.lit(SOURCE).alias("source"),
    )
    frame = frame.select(list(tables.PRICES_SCHEMA)).sort("date").unique("date", keep="last")
    frame = frame.sort("date")
    return frame.with_columns(
        pl.when(pl.col("volume") > 0).then(pl.col("volume")).otherwise(None).alias("volume")
    )


def verify_adjustment(frame: pl.DataFrame, check: AdjustmentCheck) -> bool:
    """DQ-06: an adjusted close sits clearly below the close printed at the time."""
    row = frame.filter(pl.col("date") == check.day)
    if row.is_empty():
        raise ValueError(f"{check.symbol} has no bar on {check.day}")
    return bool(row.item(0, "close") < 0.95 * check.unadjusted_close)


def ingest(
    stooq_dir: Path,
    *,
    symbols: Sequence[str],
    adjusted: Mapping[str, bool],
    checks: Sequence[AdjustmentCheck],
    snapshot: str,
    raw_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
) -> dict[str, int]:
    """Convert the universe once. The source files stay on the snapshot drive."""
    index = build_index(stooq_dir, interim_dir / SOURCE / "index.json")
    missing = [symbol for symbol in symbols if symbol not in index]
    if missing:
        raise ValueError(f"not in the snapshot: {missing}")
    rows: dict[str, int] = {}
    frames: dict[str, pl.DataFrame] = {}
    for symbol in symbols:
        frame = parse_file(index[symbol], symbol, adjusted=adjusted.get(symbol, False))
        tables.write_table(frame, tables.prices_path(processed_dir, SOURCE, symbol))
        tables.append_flags(interim_dir, SOURCE, tables.price_flags(frame, symbol))
        frames[symbol] = frame
        rows[symbol] = frame.height
    verified = {
        check.symbol: verify_adjustment(frames[check.symbol], check)
        for check in checks
        if check.symbol in frames
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        raw_dir,
        source=SOURCE,
        files=[],
        row_counts=rows,
        units="price in quote currency",
        origin=str(stooq_dir),
        release=snapshot,
        notes={"snapshot_date": snapshot, "adjustment_verified": verified},
    )
    return rows
