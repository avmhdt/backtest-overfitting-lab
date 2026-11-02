"""How many published anomalies survive multiple-testing corrections (FR-28, DQ-11)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats

import overfitlab as ol
from overfitlab_study.data import tables
from overfitlab_study.reports import GRID, INK, MUTED, SERIES_1, SERIES_2, SURFACE, write_table

F64 = NDArray[np.float64]
PERIODS_PER_YEAR = 12
MIN_MONTHS = 60
ALPHA = 0.05
HLZ_HURDLE = 3.0
WINDOWS = ("all", "in_sample", "post_publication")


def bhy_rejections(pvalues: F64, q: float = ALPHA) -> int:
    """Rejections of the Benjamini-Yekutieli step-up procedure."""
    m = pvalues.size
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    ordered = np.sort(pvalues)
    passing = np.flatnonzero(ordered <= np.arange(1, m + 1) / (m * c_m) * q)
    return int(passing[-1]) + 1 if passing.size else 0


def in_window(frame: pl.DataFrame, window: str, docs: pl.DataFrame) -> pl.DataFrame:
    """Rows of a window. Post-publication means after the publication year."""
    if window == "all":
        return frame
    if window == "in_sample":
        return frame.filter(
            pl.col("sample_start").is_not_null(),
            pl.col("date") >= pl.col("sample_start"),
            pl.col("date") <= pl.col("sample_end"),
        )
    joined = frame.join(docs.select("signal", "published"), on="signal", how="left")
    return joined.filter(pl.col("published").is_not_null()).filter(
        pl.col("date").dt.year() > pl.col("published")
    )


def wide_matrix(frame: pl.DataFrame) -> tuple[F64, list[str]]:
    """``T x N`` returns, with 0.0 for months a predictor has no portfolio in this window.

    A zero leaves a predictor's t-statistic as it is on its own months: the mean and the standard
    deviation shrink in step, so the stepdown compares like with like.
    """
    wide = frame.pivot(on="signal", index="date", values="ret").sort("date")
    names = [name for name in wide.columns if name != "date"]
    matrix = wide.select(names).fill_null(0.0).to_numpy().astype(np.float64)
    return matrix, names


def summarise(frame: pl.DataFrame, window: str, *, n_boot: int, seed: int) -> dict[str, Any]:
    per_signal = (
        frame.group_by("signal")
        .agg(
            pl.col("ret").mean().alias("mean"),
            pl.col("ret").std().alias("std"),
            pl.col("ret").len().alias("n"),
        )
        .filter(pl.col("n") >= MIN_MONTHS, pl.col("std") > 0.0)
        .sort("signal")
    )
    kept = per_signal.get_column("signal").to_list()
    tstats = (
        per_signal.get_column("mean")
        / per_signal.get_column("std")
        * per_signal.get_column("n").cast(pl.Float64).sqrt()
    ).to_numpy()
    counts = per_signal.get_column("n").to_numpy()
    pvalues = np.asarray(2.0 * stats.t.sf(np.abs(tstats), counts - 1), dtype=np.float64)
    matrix, names = wide_matrix(frame.filter(pl.col("signal").is_in(kept)))
    stepm = ol.romano_wolf_stepdown(matrix, n_boot=n_boot, rng=seed)
    return {
        "window": window,
        "n_predictors": len(names),
        "t_above_1_96": int((tstats > 1.96).sum()),
        "t_above_3": int((tstats > HLZ_HURDLE).sum()),
        "after_bhy": bhy_rejections(pvalues),
        "after_stepm": int(stepm.rejected.sum()),
    }


def best_predictor(frame: pl.DataFrame) -> dict[str, Any]:
    """DSR of the predictor with the highest Sharpe ratio, against all the others."""
    pieces = frame.partition_by("signal", maintain_order=True)
    sharpes = {
        str(piece.item(0, "signal")): float(ol.sharpe_ratio(piece.get_column("ret").to_numpy()))
        for piece in pieces
        if piece.height >= MIN_MONTHS
    }
    name = max(sharpes, key=lambda key: sharpes[key])
    returns = frame.filter(pl.col("signal") == name).get_column("ret").to_numpy()
    moments = ol.return_moments(returns)
    variance = float(np.var(list(sharpes.values()), ddof=1))
    result = ol.deflated_sharpe_ratio(
        moments.sr,
        moments.n_obs,
        n_trials=float(len(sharpes)),
        var_trials=variance,
        skew=moments.skew,
        kurtosis=moments.kurtosis,
    )
    return {
        "best_signal": name,
        "best_sr_annual": moments.sr * math.sqrt(PERIODS_PER_YEAR),
        "best_dsr": result.dsr,
        "sr0_annual": result.sr0 * math.sqrt(PERIODS_PER_YEAR),
    }


def plot(rows: Sequence[dict[str, Any]], path: Path) -> None:
    labels = {
        "t_above_1_96": "t > 1.96",
        "t_above_3": "t > 3.0",
        "after_bhy": "After BHY",
        "after_stepm": "After StepM",
    }
    by_window = {row["window"]: row for row in rows}
    series = (
        ("all", "Full history", SERIES_1),
        ("post_publication", "After publication", SERIES_2),
    )
    names = list(labels)
    fig, ax = plt.subplots(figsize=(7.2, 3.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    positions = np.arange(len(names))[::-1].astype(float)
    for offset, (window, label, colour) in zip((0.19, -0.19), series, strict=True):
        values = [by_window[window][name] for name in names]
        ax.barh(positions + offset, values, height=0.34, color=colour, label=label)
        for position, value in zip(positions + offset, values, strict=True):
            ax.annotate(
                str(value),
                (value, position),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                color=INK,
                fontsize=9,
            )
    ax.set_yticks(positions, [labels[name] for name in names], color=INK)
    totals = [by_window[window]["n_predictors"] for window, _, _ in series]
    ax.set_xlabel(f"Predictors, out of {totals[0]} and {totals[1]}", color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=MUTED, fontsize=9, loc="lower right")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def run(
    *, release: str, processed_dir: Path, reports_dir: Path, n_boot: int = 2000, seed: int = 0
) -> list[dict[str, Any]]:
    frame = pl.read_parquet(tables.signal_path(processed_dir, release))
    docs = pl.read_parquet(processed_dir / "signal_docs" / f"release={release}" / "part-0.parquet")
    rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        part = in_window(frame, window, docs)
        row = summarise(part, window, n_boot=n_boot, seed=seed)
        row.update(best_predictor(part))
        rows.append(row)
    write_table(rows, reports_dir / "tables" / "case_osap.csv")
    plot(rows, reports_dir / "figures" / "osap_hurdles.png")
    return rows
