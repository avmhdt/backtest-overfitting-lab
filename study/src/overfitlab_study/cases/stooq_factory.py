"""The strategy factory on the ETF, index and FX universe, with this repo's own trial log.

A trial is one factory variant applied to the whole universe, equal weighted. All in-sample
trials are logged first, and only then is the selected one evaluated on the holdout (DQ-08).
The holdout runs once: if the log already holds it, it is read back and never recomputed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import overfitlab as ol
from overfitlab_study.cases.universe import aligned_excess_closes, resolve
from overfitlab_study.reports import write_table

F64 = NDArray[np.float64]
FAMILY = "stooq_factory"
HOLDOUT_FAMILY = "stooq_factory_holdout"
WARMUP_BARS = 250
DATASET = "d05_universe"


def strategy_matrix(closes: F64, specs: Sequence[ol.StrategySpec], cost_bps: float) -> F64:
    """Equal-weighted net returns of each variant across the series, ``T x N``."""
    total = np.zeros((closes.shape[0] - 1, len(specs)))
    for k in range(closes.shape[1]):
        total += ol.run_strategies(closes[:, k], specs, cost_bps=cost_bps)
    return np.asarray(total / closes.shape[1], dtype=np.float64)


def holdout_start(last_day: date, holdout_years: int) -> date:
    """First day of the last ``holdout_years`` calendar years."""
    return date(last_day.year - holdout_years + 1, 1, 1)


def logged_rows(log_path: Path, family: str) -> list[Mapping[str, object]]:
    if not log_path.exists():
        return []
    log = ol.read_trial_log(log_path, validate=False)
    return [row for row in log.rows if row["strategy_family"] == family]


def log_in_sample(
    trials: ol.TrialLogger,
    specs: Sequence[ol.StrategySpec],
    dates: NDArray[np.datetime64],
    matrix: F64,
    cost_bps: float,
) -> None:
    for j, spec in enumerate(specs):
        trials.log(
            strategy_family=FAMILY,
            params=_params(spec, cost_bps),
            dates=dates,
            returns=matrix[:, j],
            extra={"x_dataset": DATASET, "x_is_holdout": "false", "x_returns_basis": "excess"},
        )


def _params(spec: ol.StrategySpec, cost_bps: float) -> dict[str, object]:
    return {
        "family": spec.family,
        **spec.params,
        "weights": "equal",
        "cost_bps": cost_bps,
        "universe": DATASET,
    }


def run(
    *,
    universe: Mapping[str, Any],
    grid: Mapping[str, Mapping[str, Sequence[object]]],
    cost_bps: float,
    holdout_years: int,
    processed_dir: Path,
    results_dir: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    log_path = results_dir / "trial_log.csv"
    trials = ol.TrialLogger(log_path, results_dir / "runs")
    dates, closes, _ = aligned_excess_closes(resolve(universe, processed_dir), processed_dir)
    specs = ol.strategy_grid(grid)
    matrix = strategy_matrix(closes, specs, cost_bps)[WARMUP_BARS:]
    days = dates[1:][WARMUP_BARS:]

    split = np.datetime64(holdout_start(days[-1].astype(object), holdout_years))
    in_sample = days < split
    if not logged_rows(log_path, FAMILY):
        log_in_sample(trials, specs, days[in_sample], matrix[in_sample], cost_bps)

    report = ol.evaluate_trial_log(log_path, families=[FAMILY])
    log = ol.read_trial_log(log_path).filter(families=[FAMILY])
    best = log.rows[log.selected_index(report.selected_run_id)]

    if not logged_rows(log_path, HOLDOUT_FAMILY):
        chosen = next(j for j, s in enumerate(specs) if _matches(s, best, cost_bps))
        trials.log(
            strategy_family=HOLDOUT_FAMILY,
            params={**_params(specs[chosen], cost_bps), "selected_run_id": report.selected_run_id},
            dates=days[~in_sample],
            returns=matrix[~in_sample, chosen],
            extra={"x_dataset": DATASET, "x_is_holdout": "true", "x_returns_basis": "excess"},
        )
    holdout = logged_rows(log_path, HOLDOUT_FAMILY)[-1]

    row = {
        "family": FAMILY,
        "n_trials": report.n_trials,
        "n_obs": report.n_obs,
        "start": str(best["start"]),
        "end": str(best["end"]),
        "selected_params": str(best["params_json"]),
        "naive_sr_annual": report.sr_annual,
        "var_trials_annual": report.variance_annual,
        "sr0_annual": report.sr0_annual,
        "dsr": report.dsr,
        "pbo": report.pbo,
        "spa_pvalue": report.spa_pvalue,
        "stepm_rejects_selected": report.stepm_rejects_selected,
        "haircut_sr_annual": report.haircut_sr_annual,
        "verdict": report.verdict,
        "holdout_start": str(holdout["start"]),
        "holdout_end": str(holdout["end"]),
        "holdout_n_obs": int(str(holdout["n_obs"])),
        "holdout_sr_annual": float(str(holdout["sharpe_annual"])),
    }
    write_table([row], reports_dir / "tables" / "case_stooq.csv")
    return row


def _matches(spec: ol.StrategySpec, row: Mapping[str, object], cost_bps: float) -> bool:
    return ol.canonical_params_json(_params(spec, cost_bps)) == row["params_json"]
