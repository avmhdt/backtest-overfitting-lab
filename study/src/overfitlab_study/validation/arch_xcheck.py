"""Cross-check of the snooping tests against the independent arch package (AC-W-07).

arch works with losses, so a benchmark of zero and losses of ``-rel_perf`` give the relative
performance used here. arch 8 compares raw means with no floor at zero, even with
``studentize=True``, where the flag only sets the screening threshold of the consistent
p-value. This library follows Hansen (2005) and studentises by default. So the p-values are
compared like with like, through ``spa_test(studentize=False)``, and the gap to the
studentised statistic is reported separately, not hidden.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from arch.bootstrap import SPA, RealityCheck, StationaryBootstrap, StepM
from numpy.typing import NDArray

import overfitlab as ol

F64 = NDArray[np.float64]
BLOCK = 5
PVALUE_TOLERANCE = 0.02
IDENTICAL_TOLERANCE = 1e-8
MAX_WORKERS = 8
TYPES = ("lower", "consistent", "upper")


@dataclass(frozen=True)
class Case:
    case_id: int
    n_strategies: int
    n_obs: int
    planted: bool


def fixed_cases() -> list[Case]:
    """Twenty cases: N in {5, 50} by T in {250, 1000} by null and planted, repeated."""
    grid = [(n, t, planted) for n in (5, 50) for t in (250, 1000) for planted in (False, True)]
    specs = (grid * 3)[:20]
    return [Case(i, n, t, planted) for i, (n, t, planted) in enumerate(specs)]


def case_data(case: Case) -> F64:
    rng = np.random.default_rng(20261030 + case.case_id)
    if case.planted:
        synth = ol.planted_signal_matrix(
            case.n_obs, case.n_strategies, true_sharpe_annual=4.0, n_true=2, rng=rng
        )
        return synth.returns
    return ol.gaussian_noise(case.n_obs, case.n_strategies, rng=rng)


def arch_indices(n_obs: int, n_boot: int, seed: int) -> NDArray[np.int64]:
    """The resamples arch draws for this size and seed, from a twin sampler."""
    sampler = StationaryBootstrap(BLOCK, np.arange(n_obs), seed=seed)
    draws = [data[0][0] for data in sampler.bootstrap(n_boot)]
    return np.asarray(draws, dtype=np.int64)


def run_case(case: Case, n_boot: int) -> dict[str, Any]:
    d = case_data(case)
    seed = 1000 + case.case_id
    losses = -d
    zero = np.zeros(d.shape[0])

    theirs_rc = RealityCheck(zero, losses, block_size=BLOCK, reps=n_boot, seed=seed)
    theirs_rc.compute()
    theirs_spa = SPA(zero, losses, block_size=BLOCK, reps=n_boot, seed=seed + 1)
    theirs_spa.compute()
    theirs_step = StepM(zero, losses, size=0.05, block_size=BLOCK, reps=n_boot, seed=seed + 2)
    theirs_step.compute()

    # Independent draws, seeded apart from arch's.
    ours_rc = ol.reality_check(d, n_boot=n_boot, block_length=BLOCK, rng=seed + 3)
    raw_spa = ol.spa_test(d, n_boot=n_boot, block_length=BLOCK, rng=seed + 4, studentize=False)
    ours_spa = ol.spa_test(d, n_boot=n_boot, block_length=BLOCK, rng=seed + 4)
    ours_step = ol.romano_wolf_stepdown(d, n_boot=n_boot, block_length=BLOCK, rng=seed + 5)

    # The same resamples arch used.
    same_rc = ol.reality_check(
        d, indices=arch_indices(d.shape[0], n_boot, seed), block_length=BLOCK
    )
    same_spa = ol.spa_test(
        d,
        indices=arch_indices(d.shape[0], n_boot, seed + 1),
        block_length=BLOCK,
        studentize=False,
    )

    their_set = sorted(int(cast(int, i)) for i in theirs_step.superior_models)
    our_set = np.flatnonzero(ours_step.rejected).tolist()
    row: dict[str, Any] = {
        "case": case.case_id,
        "n_strategies": case.n_strategies,
        "n_obs": case.n_obs,
        "planted": case.planted,
        "n_boot": n_boot,
        "rc_ours": ours_rc.pvalue,
        "rc_arch": float(theirs_rc.pvalues["upper"]),
        "rc_same_indices": same_rc.pvalue,
    }
    for kind in TYPES:
        row[f"spa_{kind}_arch"] = float(theirs_spa.pvalues[kind])
        row[f"spa_{kind}_ours_raw"] = getattr(raw_spa, f"pvalue_{kind}")
        row[f"spa_{kind}_same_indices"] = getattr(same_spa, f"pvalue_{kind}")
        row[f"spa_{kind}_ours_studentised"] = getattr(ours_spa, f"pvalue_{kind}")
    row["stepm_ours"] = " ".join(str(i) for i in our_set)
    row["stepm_arch"] = " ".join(str(i) for i in their_set)
    row["stepm_match"] = our_set == their_set
    return row


def run_all(n_boot: int) -> list[dict[str, Any]]:
    cases = fixed_cases()
    workers = min(MAX_WORKERS, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_case, cases, [n_boot] * len(cases)))


def _largest_gap(rows: Sequence[dict[str, Any]], pairs: Sequence[tuple[str, str]]) -> float:
    return float(max(abs(row[a] - row[b]) for row in rows for a, b in pairs))


def check_criteria(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    like_for_like = [("rc_ours", "rc_arch")]
    like_for_like += [(f"spa_{k}_ours_raw", f"spa_{k}_arch") for k in TYPES]
    identical = [("rc_same_indices", "rc_arch")]
    identical += [(f"spa_{k}_same_indices", f"spa_{k}_arch") for k in TYPES]
    studentised = [(f"spa_{k}_ours_studentised", f"spa_{k}_arch") for k in TYPES]
    worst = _largest_gap(rows, like_for_like)
    same = _largest_gap(rows, identical)
    matches = sum(row["stepm_match"] for row in rows)
    return {
        "pvalues_within_tolerance": bool(worst <= PVALUE_TOLERANCE),
        "max_pvalue_diff": float(worst),
        "stepm_sets_match": bool(matches >= 19),
        "stepm_matches": int(matches),
        "identical_indices_agree": bool(same <= IDENTICAL_TOLERANCE),
        "max_identical_diff": float(same),
        "max_studentised_gap": float(_largest_gap(rows, studentised)),
    }


def passed(criteria: dict[str, Any]) -> bool:
    return bool(
        criteria["pvalues_within_tolerance"]
        and criteria["stepm_sets_match"]
        and criteria["identical_indices_agree"]
    )


def write_report(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in row.items()})
