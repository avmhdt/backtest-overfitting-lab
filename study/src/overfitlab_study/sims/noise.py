"""The best of many strategies on pure noise (FR-21, AC-W-01)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from numpy.typing import NDArray

import overfitlab as ol
from overfitlab_study.config import experiment_seeds
from overfitlab_study.reports import GRID, MUTED, SERIES_1, SERIES_2, SURFACE, write_table

F64 = NDArray[np.float64]
SETTINGS = ("independent", "factory_gbm")
LABELS = {"independent": "1,000 independent", "factory_gbm": "Factory on GBM"}
MAX_WORKERS = 8
QUICK_STRATEGIES = 200
QUICK_SEEDS = 20
PERIODS_PER_YEAR = 252


@dataclass(frozen=True)
class NoiseConfig:
    seeds: int
    n_obs: int
    vol_annual: float
    n_strategies: int
    pbo_partitions: int
    n_boot: int
    cost_bps: float
    quick: bool = False

    @classmethod
    def from_mapping(
        cls, config: Mapping[str, Any], cost_bps: float, quick: bool = False
    ) -> NoiseConfig:
        return cls(
            seeds=QUICK_SEEDS if quick else config["seeds"],
            n_obs=config["n_obs"],
            vol_annual=config["vol_annual"],
            n_strategies=QUICK_STRATEGIES if quick else config["settings"][0]["n_strategies"],
            pbo_partitions=config["pbo_partitions"],
            n_boot=config["n_boot"],
            cost_bps=cost_bps,
            quick=quick,
        )


def strategy_returns(setting: str, config: NoiseConfig, rng: np.random.Generator) -> F64:
    if setting == "independent":
        return ol.gaussian_noise(
            config.n_obs, config.n_strategies, vol_annual=config.vol_annual, rng=rng
        )
    prices = ol.gbm_prices(config.n_obs + 1, vol_annual=config.vol_annual, rng=rng)
    grid = ol.strategy_grid()
    if config.quick:
        grid = grid[:: len(grid) // QUICK_STRATEGIES]
    return ol.run_strategies(prices, grid, cost_bps=config.cost_bps)


def run_seed(setting: str, config: NoiseConfig, seed: np.random.SeedSequence) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    returns = strategy_returns(setting, config, rng)
    gate = ol.VerdictConfig(pbo_partitions=config.pbo_partitions, n_boot=config.n_boot)
    report = ol.evaluate(returns, config=gate, rng=rng)
    best = returns[:, report.selected]
    assert report.pbo is not None
    assert report.spa is not None
    return {
        "setting": setting,
        "n_trials": returns.shape[1],
        "best_sr_annual": ol.annualize_sharpe(report.dsr.sr, PERIODS_PER_YEAR),
        "naive_psr": ol.psr_from_returns(best),
        "dsr": report.dsr.dsr,
        "pbo": report.pbo.pbo,
        "rc_pvalue": ol.reality_check(returns, n_boot=config.n_boot, rng=rng).pvalue,
        "spa_pvalue": report.spa.pvalue_consistent,
        "verdict": report.verdict,
    }


def summarise(outcomes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for setting in SETTINGS:
        part = [item for item in outcomes if item["setting"] == setting]
        rows.append(
            {
                "setting": setting,
                "n_seeds": len(part),
                "n_trials": part[0]["n_trials"],
                "median_best_sr_annual": median(i["best_sr_annual"] for i in part),
                "seeds_naive_psr_95": sum(i["naive_psr"] >= 0.95 for i in part),
                "seeds_dsr_below_95": sum(i["dsr"] < 0.95 for i in part),
                "median_dsr": median(i["dsr"] for i in part),
                "median_pbo": median(i["pbo"] for i in part),
                "median_rc_pvalue": median(i["rc_pvalue"] for i in part),
                "median_spa_pvalue": median(i["spa_pvalue"] for i in part),
                "seeds_pass": sum(i["verdict"] == "PASS" for i in part),
                "seeds_fail": sum(i["verdict"] == "FAIL" for i in part),
                "seeds_inconclusive": sum(i["verdict"] == "INCONCLUSIVE" for i in part),
            }
        )
    return rows


def check_criteria(rows: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    """AC-W-01, with the seed counts scaled to the number of seeds that ran."""
    independent, factory = rows
    needed = {row["setting"]: 0.95 * row["n_seeds"] for row in rows}
    return {
        "best_sr_at_least_1_2": bool(independent["median_best_sr_annual"] >= 1.2),
        "naive_psr_significant": bool(independent["seeds_naive_psr_95"] >= needed["independent"]),
        "dsr_not_significant_independent": bool(
            independent["seeds_dsr_below_95"] >= needed["independent"]
        ),
        "pbo_at_least_0_4_independent": bool(independent["median_pbo"] >= 0.40),
        "dsr_not_significant_factory": bool(factory["seeds_dsr_below_95"] >= needed["factory_gbm"]),
        "pbo_at_least_0_4_factory": bool(factory["median_pbo"] >= 0.40),
    }


def plot(outcomes: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for setting, color, marker in zip(SETTINGS, (SERIES_1, SERIES_2), ("o", "s"), strict=True):
        part = [item for item in outcomes if item["setting"] == setting]
        ax.scatter(
            [i["best_sr_annual"] for i in part],
            [i["dsr"] for i in part],
            s=20,
            color=color,
            marker=marker,
            alpha=0.7,
            linewidths=0,
            label=LABELS[setting],
        )
    ax.axhline(0.95, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(
        "Significance at 0.95",
        (1.0, 0.95),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        color=MUTED,
        fontsize=9,
    )
    ax.set_xlabel("Best naive annual Sharpe ratio (one point per seed)", color=MUTED)
    ax.set_ylabel("Deflated Sharpe Ratio", color=MUTED)
    ax.set_ylim(-0.03, 1.03)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, loc="lower right", labelcolor=MUTED, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def run(
    config: NoiseConfig,
    *,
    seeds: int,
    master_seed: int,
    reports_dir: Path,
    results_dir: Path,
    config_key: str,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    jobs: list[tuple[str, np.random.SeedSequence]] = [
        (setting, sequence)
        for setting in SETTINGS
        for sequence in experiment_seeds(master_seed, f"noise-{setting}", seeds)
    ]
    workers = min(MAX_WORKERS, os.cpu_count() or 1, len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outcomes = list(
            pool.map(run_seed, [j[0] for j in jobs], [config] * len(jobs), [j[1] for j in jobs])
        )

    rows = summarise(outcomes)
    write_table(rows, reports_dir / "tables" / "noise_experiment.csv")
    plot(outcomes, reports_dir / "figures" / "noise_best_vs_dsr.png")
    target = results_dir / "sims" / "noise" / f"{config_key}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(outcomes).write_parquet(target, compression="zstd")
    return rows, check_criteria(rows)
