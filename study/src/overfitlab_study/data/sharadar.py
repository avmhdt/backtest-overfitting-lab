"""U.S. ETF prices from Sharadar SFP, through Sharadar's direct API.

The key goes in the ``x-api-key`` header, so it stays out of URLs and logs. Raw pulls stay
under the data root and are never committed.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from overfitlab_study.data import tables
from overfitlab_study.data.manifest import write_manifest
from overfitlab_study.settings import Settings

SOURCE = "sharadar"
FUNDS_URL = "https://api.sharadar.com/v1.0/data/funds"
PAGE_SIZE = 10_000
START = "1993-01-01"
FIELDS = "ticker,date,open,high,low,close,volume,closeadj,closeunadj"


class MissingKeyError(RuntimeError):
    """SHARADAR_API_KEY is not set."""


def api_key(settings: Settings) -> str:
    if settings.sharadar_api_key is None:
        raise MissingKeyError(
            "SHARADAR_API_KEY is not set; U.S. ETF prices fall back to the local Stooq snapshot"
        )
    return settings.sharadar_api_key.get_secret_value()


def fetch(symbols: Sequence[str], key: str, client: httpx.Client) -> str:
    """Page through SFP and return one CSV text with a single header."""
    header = ""
    rows: list[str] = []
    skip = 0
    while True:
        params: dict[str, str | int] = {
            "ticker": ",".join(symbols),
            "from": START,
            "fields": FIELDS,
            "sort": "ticker.asc,date.asc",
            "limit": PAGE_SIZE,
            "skip": skip,
            "format": "csv",
        }
        response = client.get(FUNDS_URL, params=params, headers={"x-api-key": key}, timeout=120.0)
        response.raise_for_status()
        lines = response.text.splitlines()
        if not lines:
            return "\n".join([header, *rows])
        header = lines[0]
        rows += lines[1:]
        if len(lines) - 1 < PAGE_SIZE:
            return "\n".join([header, *rows]) + "\n"
        skip += PAGE_SIZE


def to_prices(text: str) -> pl.DataFrame:
    """SFP to prices_daily. ``closeadj`` carries dividends, so the bar is scaled to it (DQ-06)."""
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return pl.DataFrame(schema=tables.PRICES_SCHEMA)
    raw = pl.DataFrame(rows).with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col("open", "high", "low", "close", "volume", "closeadj").cast(pl.Float64),
    )
    scale = pl.col("closeadj") / pl.col("close")
    frame = raw.select(
        "date",
        pl.col("ticker").str.to_lowercase().alias("symbol"),
        (pl.col("open") * scale).alias("open"),
        (pl.col("high") * scale).alias("high"),
        (pl.col("low") * scale).alias("low"),
        pl.col("closeadj").alias("close"),
        "volume",
        pl.lit(True).alias("adjusted"),
        pl.lit(SOURCE).alias("source"),
    )
    return frame.select(list(tables.PRICES_SCHEMA)).sort("symbol", "date")


def verify_adjustment(text: str, symbol: str, day: date) -> bool:
    """DQ-06: on an old date ``closeadj`` sits below the split-adjusted close."""
    for row in csv.DictReader(io.StringIO(text)):
        if row["ticker"].lower() == symbol and row["date"] == day.isoformat():
            return float(row["closeadj"]) < 0.95 * float(row["close"])
    raise ValueError(f"{symbol} has no bar on {day}")


def crosscheck(tickers: Mapping[str, str], processed_dir: Path) -> list[dict[str, Any]]:
    """Daily return agreement with the Stooq snapshot, per ETF. Aggregates only."""
    rows: list[dict[str, Any]] = []
    for ticker, stooq_ticker in tickers.items():
        ours = tables.prices_path(processed_dir, SOURCE, ticker)
        theirs = tables.prices_path(processed_dir, "stooq", stooq_ticker)
        if not ours.exists() or not theirs.exists():
            continue
        joined = (
            pl.read_parquet(ours)
            .select("date", pl.col("close").alias("a"))
            .join(pl.read_parquet(theirs).select("date", pl.col("close").alias("b")), on="date")
            .sort("date")
            .with_columns(pl.col("a").pct_change(), pl.col("b").pct_change())
            .drop_nulls()
        )
        rows.append(
            {
                "ticker": ticker,
                "common_days": joined.height,
                "return_correlation": float(joined.select(pl.corr("a", "b")).item()),
                "mean_abs_diff": float(
                    joined.select((pl.col("a") - pl.col("b")).abs().mean()).item()
                ),
            }
        )
    return rows


def ingest(
    symbols: Sequence[str],
    settings: Settings,
    client: httpx.Client,
    *,
    check_symbol: str,
    check_day: date,
) -> dict[str, int]:
    key = api_key(settings)
    text = fetch([symbol.upper() for symbol in symbols], key, client)
    raw_dir = settings.raw_dir / SOURCE
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path: Path = raw_dir / "funds.csv"
    raw_path.write_text(text, encoding="utf-8")

    prices = to_prices(text)
    rows: dict[str, int] = {}
    for frame in prices.partition_by("symbol", maintain_order=True):
        symbol = str(frame.item(0, "symbol"))
        tables.write_table(frame, tables.prices_path(settings.processed_dir, SOURCE, symbol))
        tables.append_flags(settings.interim_dir, SOURCE, tables.price_flags(frame, symbol))
        rows[symbol] = frame.height
    write_manifest(
        raw_dir,
        source=SOURCE,
        files=[raw_path],
        row_counts=rows,
        units="USD per share, adjusted by closeadj",
        origin=FUNDS_URL,
        release="SFP",
        notes={"adjustment_verified": verify_adjustment(text, check_symbol, check_day)},
    )
    return rows
