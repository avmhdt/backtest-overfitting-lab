"""French and AQR factors: StepM survivors and haircuts across long factor histories."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import overfitlab as ol
from overfitlab_study.data import tables
from overfitlab_study.reports import write_table

PERIODS_PER_YEAR = 12
FRENCH_MONTHLY = "F-F_Research_Data_Factors"
FRENCH_FACTORS = ("Mkt-RF", "SMB", "HML")
MIN_MONTHS = 240


def load_panel(aqr_datasets: Sequence[str], processed_dir: Path) -> pl.DataFrame:
    """Wide monthly panel with one column per ``source/dataset/factor``."""
    frames = [
        pl.read_parquet(tables.factor_path(processed_dir, "french", FRENCH_MONTHLY))
        .filter(pl.col("factor").is_in(FRENCH_FACTORS), pl.col("frequency") == "M")
        .with_columns(("french/" + pl.col("factor")).alias("column"))
    ]
    for dataset in aqr_datasets:
        frame = pl.read_parquet(tables.factor_path(processed_dir, "aqr", dataset))
        frames.append(frame.with_columns((f"aqr/{dataset}/" + pl.col("factor")).alias("column")))
    long = pl.concat([f.select("date", "column", "ret") for f in frames]).drop_nulls("ret")
    return long.pivot(on="column", index="date", values="ret").sort("date")


def complete_window(panel: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """The longest stretch ending at the last month, keeping columns that span it.

    Columns stopping early would shorten the window for all, so the end is the latest month with
    at least half of the columns, and columns missing inside the window are dropped and reported.
    """
    names = [c for c in panel.columns if c != "date"]
    counts = panel.select(pl.sum_horizontal(pl.col(names).is_not_null())).to_series()
    usable = panel.filter(counts >= len(names) / 2)
    last: date = usable.get_column("date").to_list()[-1]
    kept: list[str] = []
    dropped: list[str] = []
    starts: list[date] = []
    for name in names:
        present = usable.filter(pl.col(name).is_not_null()).get_column("date").to_list()
        if len(present) >= MIN_MONTHS and present[-1] == last:
            kept.append(name)
            starts.append(present[0])
        else:
            dropped.append(name)
    window = usable.filter(pl.col("date") >= max(starts))
    return window.select("date", *kept).drop_nulls(), dropped


def run(*, aqr_datasets: Sequence[str], processed_dir: Path, reports_dir: Path) -> dict[str, Any]:
    panel, dropped = complete_window(load_panel(aqr_datasets, processed_dir))
    names = [c for c in panel.columns if c != "date"]
    matrix = panel.select(names).to_numpy().astype(np.float64)
    n_obs = matrix.shape[0]
    stepm = ol.romano_wolf_stepdown(matrix, n_boot=5000, rng=0)
    sharpes = np.atleast_1d(ol.sharpe_ratio(matrix))
    rows: list[dict[str, Any]] = []
    for j, name in enumerate(names):
        sr_annual = ol.annualize_sharpe(float(sharpes[j]), PERIODS_PER_YEAR)
        haircut = ol.haircut_sharpe(
            sr_annual,
            n_obs,
            periods_per_year=PERIODS_PER_YEAR,
            n_tests=len(names),
            method="bhy",
            rng=0,
        )
        rows.append(
            {
                "factor": name,
                "n_obs": n_obs,
                "sr_annual": sr_annual,
                "tstat": float(sharpes[j]) * math.sqrt(n_obs),
                "stepm_rejected": bool(stepm.rejected[j]),
                "stepm_adj_pvalue": float(stepm.adjusted_pvalues[j]),
                "bhy_haircut_sr_annual": haircut.haircut_sr_annual,
                "bhy_haircut_pct": haircut.haircut_pct,
            }
        )
    write_table(rows, reports_dir / "tables" / "case_factors.csv")
    return {
        "factors": len(names),
        "n_obs": n_obs,
        "start": str(panel.get_column("date").min()),
        "end": str(panel.get_column("date").max()),
        "stepm_survivors": int(stepm.rejected.sum()),
        "dropped": dropped,
    }
