"""Planted-signal power: the check against theory and the full grid (FR-22, FR-23)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from scipy import stats

import overfitlab as ol
from overfitlab_study.config import experiment_seeds
from overfitlab_study.reports import GRID, INK, MUTED, SURFACE, write_table
from overfitlab_study.sims import theory

F64 = NDArray[np.float64]
BOOT_TESTS = ("rc", "spa", "stepm", "haircut_bhy", "pbo")
# dsr estimates V from the same trials. dsr_known_v uses the null variance, which is what
# the reference assumes. It is a diagnostic and never counts toward AC-W-02.
CHECK_TESTS = ("psr", "dsr", "dsr_known_v", "rc", "spa", "stepm")
TEST_ORDER = ("psr", "dsr", "rc", "spa", "stepm", "haircut_bhy", "pbo")
LABELS = {
    "psr": "PSR of the best trial",
    "dsr": "DSR",
    "rc": "Reality Check",
    "spa": "SPA",
    "stepm": "StepM",
    "haircut_bhy": "BHY haircut",
    "pbo": "PBO at most 0.20",
}
# Blue ramp, light to dark: longer backtests read as darker.
LENGTH_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b")
PERIODS_PER_YEAR = 252
PBO_PARTITIONS = 16
PBO_MAX = 0.20
IID_BLOCK = 1.0
CHUNK = 25
MAX_WORKERS = 16
MIN_YEARS_SHARPES = (0.5, 1.0, 1.5, 2.0)
TARGET_POWER = 0.80
PSR_TOLERANCE = 0.03
MAX_STAT_TOLERANCE = 0.05
QUICK_TOLERANCE = 0.10
PBO_SPEARMAN_LIMIT = -0.9
MONOTONE_TOLERANCE = 0.03


@dataclass(frozen=True)
class Cell:
    true_sharpe_annual: float
    years: float
    n_trials: int

    @property
    def key(self) -> str:
        return f"{self.true_sharpe_annual}-{self.years}-{self.n_trials}"


@dataclass(frozen=True)
class PowerConfig:
    alpha: float
    reps_closed_form: int
    reps_bootstrap: int
    n_boot: int
    corr: float
    true_sharpes: tuple[float, ...]
    years: tuple[float, ...]
    n_trials: tuple[int, ...]
    check_cells: tuple[Cell, ...]
    tests: tuple[str, ...]

    @classmethod
    def from_mapping(cls, study: Mapping[str, Any], quick: bool = False) -> PowerConfig:
        section = study["power"]
        cells = tuple(
            Cell(c["true_sharpe_annual"], c["years"], c["n_trials"]) for c in section["check_cells"]
        )
        return cls(
            alpha=study["alpha"],
            reps_closed_form=200 if quick else section["reps_closed_form"],
            reps_bootstrap=100 if quick else section["reps_bootstrap"],
            n_boot=300 if quick else section["n_boot"],
            corr=section["corr_between_trials"],
            true_sharpes=tuple(section["true_sharpe_annual"]),
            years=tuple(section["years"]),
            n_trials=tuple(section["n_trials"]),
            check_cells=cells,
            tests=tuple(section["tests"]),
        )

    def grid(self) -> list[Cell]:
        return [
            Cell(sr, years, n)
            for sr in self.true_sharpes
            for years in self.years
            for n in self.n_trials
        ]

    def pbo_trend_cells(self) -> list[Cell]:
        """The SR grid at five years, for the PBO trend of AC-W-02."""
        return [Cell(sr, 5.0, 100) for sr in self.true_sharpes]


def _matrix(cell: Cell, corr: float, rng: np.random.Generator) -> F64:
    n_obs = round(cell.years * PERIODS_PER_YEAR)
    return ol.planted_signal_matrix(
        n_obs,
        cell.n_trials,
        true_sharpe_annual=cell.true_sharpe_annual,
        corr=corr,
        rng=rng,
    ).returns


def closed_rep(cell: Cell, config: PowerConfig, seed: np.random.SeedSequence) -> dict[str, float]:
    """One replication of the tests with no resampling."""
    returns = _matrix(cell, config.corr, np.random.default_rng(seed))
    report = ol.dsr_from_returns(returns)
    naive = ol.probabilistic_sharpe_ratio(
        report.sr, report.n_obs, skew=report.skew, kurtosis=report.kurtosis
    )
    known_v = ol.deflated_sharpe_ratio(
        report.sr,
        report.n_obs,
        n_trials=report.n_trials,
        var_trials=1.0 / (report.n_obs - 1),
        skew=report.skew,
        kurtosis=report.kurtosis,
    )
    return {
        "psr": float(naive >= 1.0 - config.alpha),
        "dsr": float(report.dsr >= 1.0 - config.alpha),
        "dsr_known_v": float(known_v.dsr >= 1.0 - config.alpha),
    }


def boot_rep(cell: Cell, config: PowerConfig, seed: np.random.SeedSequence) -> dict[str, float]:
    """One replication of the resampling tests, on shared bootstrap indices."""
    rng = np.random.default_rng(seed)
    returns = _matrix(cell, config.corr, rng)
    n_obs = returns.shape[0]
    indices = ol.stationary_bootstrap_indices(n_obs, config.n_boot, mean_block=IID_BLOCK, rng=rng)
    sharpes = np.atleast_1d(ol.sharpe_ratio(returns))
    best_annual = ol.annualize_sharpe(float(sharpes.max()), PERIODS_PER_YEAR)
    haircut = ol.haircut_sharpe(
        best_annual,
        n_obs,
        periods_per_year=PERIODS_PER_YEAR,
        n_tests=cell.n_trials,
        method="bhy",
        rho_tests=config.corr,
        n_sims=1000,
        rng=rng,
    )
    outcome = {
        "rc": float(ol.reality_check(returns, indices=indices).pvalue <= config.alpha),
        "spa": float(
            ol.spa_test(returns, indices=indices, block_length=IID_BLOCK).pvalue_consistent
            <= config.alpha
        ),
        "stepm": float(
            ol.romano_wolf_stepdown(
                returns, alpha=config.alpha, indices=indices, block_length=IID_BLOCK
            ).rejected.any()
        ),
        "haircut_bhy": float(haircut.adj_pvalue <= config.alpha),
        "pbo_value": float("nan"),
        "pbo": float("nan"),
    }
    if cell.n_trials >= 2:
        value = ol.pbo_cscv(returns, n_partitions=PBO_PARTITIONS).pbo
        outcome["pbo_value"] = value
        outcome["pbo"] = float(value <= PBO_MAX)
    return outcome


def run_chunk(
    cell: Cell, config: PowerConfig, kind: str, seeds: Sequence[np.random.SeedSequence]
) -> list[dict[str, float]]:
    step = closed_rep if kind == "closed" else boot_rep
    return [step(cell, config, seed) for seed in seeds]


def simulate(
    cells: Sequence[Cell], config: PowerConfig, master_seed: int
) -> dict[Cell, dict[str, F64]]:
    """Run every cell, in chunks across processes, with seeds fixed per cell and replication."""
    jobs: list[tuple[Cell, str, list[np.random.SeedSequence]]] = []
    for cell in cells:
        for kind, reps in (("closed", config.reps_closed_form), ("boot", config.reps_bootstrap)):
            seeds = experiment_seeds(master_seed, f"power-{kind}-{cell.key}", reps)
            jobs += [(cell, kind, seeds[i : i + CHUNK]) for i in range(0, reps, CHUNK)]
    workers = min(MAX_WORKERS, os.cpu_count() or 1, len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        chunks = list(
            pool.map(
                run_chunk,
                [j[0] for j in jobs],
                [config] * len(jobs),
                [j[1] for j in jobs],
                [j[2] for j in jobs],
            )
        )
    gathered: dict[Cell, dict[str, list[float]]] = {cell: {} for cell in cells}
    for job, outcomes in zip(jobs, chunks, strict=True):
        for outcome in outcomes:
            for name, value in outcome.items():
                gathered[job[0]].setdefault(name, []).append(value)
    return {
        cell: {name: np.array(values) for name, values in tests.items()}
        for cell, tests in gathered.items()
    }


def rejection_rate(values: F64) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def check_rows(
    results: Mapping[Cell, Mapping[str, F64]], config: PowerConfig, *, quick: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in config.check_cells:
        for test in CHECK_TESTS:
            if test == "psr" and cell.n_trials != 1:
                continue
            tolerance = PSR_TOLERANCE if test == "psr" else MAX_STAT_TOLERANCE
            tolerance = QUICK_TOLERANCE if quick else tolerance
            empirical = rejection_rate(results[cell][test])
            reference = theory.reference_power(
                "dsr" if test == "dsr_known_v" else test,
                true_sharpe_annual=cell.true_sharpe_annual,
                years=cell.years,
                n_trials=cell.n_trials,
                alpha=config.alpha,
                periods_per_year=PERIODS_PER_YEAR,
            )
            rows.append(
                {
                    "true_sharpe_annual": cell.true_sharpe_annual,
                    "years": cell.years,
                    "n_trials": cell.n_trials,
                    "test": test,
                    "reps": int(results[cell][test].size),
                    "empirical": empirical,
                    "reference": reference,
                    "tolerance": tolerance,
                    "within_tolerance": abs(empirical - reference) <= tolerance,
                    "diagnostic": test == "dsr_known_v",
                }
            )
    return rows


def pbo_trend(
    results: Mapping[Cell, Mapping[str, F64]], config: PowerConfig
) -> tuple[list[float], float]:
    """Median PBO along the SR grid, and its Spearman correlation with true SR."""
    cells = config.pbo_trend_cells()
    medians = [float(np.nanmedian(results[cell]["pbo_value"])) for cell in cells]
    rho = float(stats.spearmanr([cell.true_sharpe_annual for cell in cells], medians).statistic)
    return medians, rho


def grid_rows(
    results: Mapping[Cell, Mapping[str, F64]], config: PowerConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell, tests in results.items():
        for test in TEST_ORDER:
            if test not in config.tests or test not in tests:
                continue
            finite = tests[test][np.isfinite(tests[test])]
            if finite.size == 0:
                continue
            rows.append(
                {
                    "true_sharpe_annual": cell.true_sharpe_annual,
                    "years": cell.years,
                    "n_trials": cell.n_trials,
                    "test": test,
                    "reps": int(finite.size),
                    "power": float(finite.mean()),
                }
            )
    return rows


def min_length_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Shortest grid length reaching 80% power, or blank if none does."""
    out: list[dict[str, Any]] = []
    tests = [t for t in TEST_ORDER if any(r["test"] == t for r in rows)]
    for test in tests:
        for n_trials in sorted({r["n_trials"] for r in rows if r["test"] == test}):
            for sharpe in MIN_YEARS_SHARPES:
                matches = sorted(
                    (r["years"], r["power"])
                    for r in rows
                    if r["test"] == test
                    and r["n_trials"] == n_trials
                    and r["true_sharpe_annual"] == sharpe
                )
                reached = [years for years, power in matches if power >= TARGET_POWER]
                out.append(
                    {
                        "test": test,
                        "true_sharpe_annual": sharpe,
                        "n_trials": n_trials,
                        "min_years_for_80pct": reached[0] if reached else None,
                    }
                )
    return out


def monotone_violations(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Curves that fall by more than Monte Carlo error as true SR rises (AC-H-01)."""
    problems: list[str] = []
    keys = sorted({(r["test"], r["years"], r["n_trials"]) for r in rows})
    for test, years, n_trials in keys:
        curve = sorted(
            (r["true_sharpe_annual"], r["power"])
            for r in rows
            if (r["test"], r["years"], r["n_trials"]) == (test, years, n_trials)
        )
        powers = [power for _, power in curve]
        peak = np.maximum.accumulate(powers)
        if (peak - powers).max() > MONOTONE_TOLERANCE:
            problems.append(f"{test}, {years} years, N = {n_trials}")
    return problems


def _panel_rows(rows: Sequence[Mapping[str, Any]], test: str, n_trials: int) -> dict[float, Any]:
    lines: dict[float, Any] = {}
    for row in rows:
        if row["test"] == test and row["n_trials"] == n_trials:
            lines.setdefault(row["years"], []).append((row["true_sharpe_annual"], row["power"]))
    return {years: sorted(points) for years, points in sorted(lines.items())}


def plot(rows: Sequence[Mapping[str, Any]], path: Path, tests: Sequence[str]) -> None:
    columns = (10, 100)
    fig, axes = plt.subplots(
        len(tests), 2, figsize=(7.6, 1.75 * len(tests)), sharex=True, sharey=True, facecolor=SURFACE
    )
    handles = []
    for r, test in enumerate(tests):
        for c, n_trials in enumerate(columns):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            lines = _panel_rows(rows, test, n_trials)
            for colour, (years, points) in zip(LENGTH_RAMP, lines.items(), strict=False):
                x, y = zip(*points, strict=True)
                (line,) = ax.plot(x, y, color=colour, linewidth=1.6, label=f"{years:g} y")
                if r == 0 and c == 0:
                    handles.append(line)
            ax.axhline(TARGET_POWER, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
            ax.set_ylim(-0.03, 1.03)
            ax.grid(color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(colors=MUTED, length=0, labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(f"N = {n_trials}", color=INK, fontsize=10)
            if c == 0:
                ax.set_ylabel(LABELS[test], color=INK, fontsize=8)
            if r == len(tests) - 1:
                ax.set_xlabel("True annual Sharpe ratio", color=MUTED, fontsize=9)
    fig.legend(
        handles=handles,
        title="Backtest length",
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        labelcolor=MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_html(rows: Sequence[Mapping[str, Any]], path: Path, tests: Sequence[str]) -> None:
    figure = make_subplots(
        rows=len(tests),
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        column_titles=["N = 10", "N = 100"],
        row_titles=[LABELS[test] for test in tests],
    )
    for r, test in enumerate(tests, start=1):
        for c, n_trials in enumerate((10, 100), start=1):
            lines = _panel_rows(rows, test, n_trials)
            for colour, (years, points) in zip(LENGTH_RAMP, lines.items(), strict=False):
                x, y = zip(*points, strict=True)
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=f"{years:g} y",
                        legendgroup=f"{years:g}",
                        showlegend=r == 1 and c == 1,
                        line={"color": colour},
                    ),
                    row=r,
                    col=c,
                )
    figure.update_yaxes(range=[-0.03, 1.03])
    figure.update_layout(height=190 * len(tests), template="simple_white")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs="cdn", full_html=True, div_id="power-curves")


def run_check(
    config: PowerConfig, *, master_seed: int, reports_dir: Path, quick: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """AC-W-02: the check cells against theory, and the PBO trend."""
    cells = list(dict.fromkeys([*config.check_cells, *config.pbo_trend_cells()]))
    results = simulate(cells, config, master_seed)
    rows = check_rows(results, config, quick=quick)
    medians, rho = pbo_trend(results, config)
    write_table(rows, reports_dir / "tables" / "power_check.csv")
    criteria = {
        "all_within_tolerance": all(
            row["within_tolerance"] for row in rows if not row["diagnostic"]
        ),
        "pbo_spearman": rho,
        "pbo_trend_ok": bool(rho <= PBO_SPEARMAN_LIMIT),
        "pbo_medians": medians,
    }
    return rows, criteria


def run_grid(
    config: PowerConfig, *, master_seed: int, reports_dir: Path, docs_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """AC-H-01: the full grid, curves and minimum lengths."""
    results = simulate(config.grid(), config, master_seed)
    rows = grid_rows(results, config)
    write_table(rows, reports_dir / "tables" / "power_grid.csv")
    write_table(min_length_rows(rows), reports_dir / "tables" / "min_length_for_power.csv")
    tests = [t for t in TEST_ORDER if t in config.tests]
    plot(rows, reports_dir / "figures" / "power_curves.png", tests)
    plot_html(rows, docs_dir / "assets" / "power_curves.html", tests)
    violations = monotone_violations(rows)
    return rows, {"curves_monotone": not violations, "violations": violations}
