"""Loading the case-study universe on one aligned calendar."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import yaml
from numpy.typing import NDArray

from overfitlab_study.data import tables

UNIVERSE = Path("experiments/universe.yaml")
RF_DATASET = "F-F_Research_Data_Factors_daily"
F64 = NDArray[np.float64]


def load_universe(path: Path = UNIVERSE) -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Series:
    name: str
    path: Path
    excess_of_rf: bool


def resolve(universe: Mapping[str, Any], processed_dir: Path) -> list[Series]:
    """ETFs from Sharadar, or from Stooq as the fallback; FX and the index from Stooq."""
    items: list[Series] = []
    for ticker, stooq_ticker in universe["etfs"].items():
        path = tables.prices_path(processed_dir, "sharadar", ticker)
        if not path.exists():
            path = tables.prices_path(processed_dir, "stooq", stooq_ticker)
        items.append(Series(ticker, path, excess_of_rf=True))
    for symbol in universe["index"]:
        items.append(Series(symbol, tables.prices_path(processed_dir, "stooq", symbol), True))
    for symbol in universe["fx"]:
        items.append(Series(symbol, tables.prices_path(processed_dir, "stooq", symbol), False))
    return items


def risk_free(processed_dir: Path) -> pl.DataFrame:
    """Daily decimal risk-free rate from the Ken French library (DQ-07)."""
    path = tables.factor_path(processed_dir, "french", RF_DATASET)
    frame = pl.read_parquet(path).filter(pl.col("factor") == "RF")
    return frame.select("date", pl.col("ret").alias("rf")).drop_nulls()


def aligned_excess_closes(
    series: list[Series], processed_dir: Path
) -> tuple[NDArray[np.datetime64], F64, list[str]]:
    """Excess-return price paths on the dates all series share (DQ-01, DQ-07).

    Dates are the inner join of every series and the risk-free rate. FX is already excess, as a
    funded position in the pair earns the spread, so it is left as it is.
    """
    joined = risk_free(processed_dir)
    for item in series:
        frame = pl.read_parquet(item.path).select("date", pl.col("close").alias(item.name))
        joined = joined.join(frame, on="date", how="inner")
    joined = joined.sort("date")
    rf = joined.get_column("rf").to_numpy()
    closes = np.ones((joined.height, len(series)))
    for k, item in enumerate(series):
        values = joined.get_column(item.name).to_numpy()
        returns = values[1:] / values[:-1] - 1.0
        if item.excess_of_rf:
            returns = returns - rf[1:]
        closes[1:, k] = np.cumprod(1.0 + returns)
    dates = joined.get_column("date").to_numpy().astype("datetime64[D]")
    return dates, closes, [item.name for item in series]
