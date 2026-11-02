"""Write tests/fixtures/js_cases.json from the Python implementation.

Every expected value comes from overfitlab or scipy, so the calculator's tests compare it
with the library and not with itself. Run with ``uv run python scripts/gen_js_cases.py``.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri, stdtr, stdtrit

import overfitlab as ol

SEED = 20261102
N_SCENARIOS = 100
OUTPUT = Path(__file__).parents[1] / "tests" / "fixtures" / "js_cases.json"

GRID: dict[str, list[float]] = {
    "sr_annual": [-0.5, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0],
    "years": [1, 2, 3, 5, 10, 20],
    "periods_per_year": [252, 365, 52, 12],
    "skew": [0.0, 0.5, -0.5, -1.0, -3.0],
    "kurtosis": [2.0, 3.0, 5.0, 10.0],
    "n_trials": [1, 2, 10, 100, 1000, 1000000],
    "var_trials_annual": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
    "n_tests": [1, 2, 10, 100, 1000],
    "autocorr": [0.0, 0.1, 0.3, -0.2],
}

NAMED: dict[str, dict[str, Any]] = {
    "dsr_paper": {
        "sr_annual": 2.5,
        "n_obs": 1250,
        "periods_per_year": 250,
        "skew": -3.0,
        "kurtosis": 10.0,
        "n_trials": 100,
        "var_trials_annual": 0.5,
        "n_tests": 100,
        "autocorr": 0.0,
    },
    "harvey_liu": {
        "sr_annual": 1.0,
        "n_obs": 120,
        "periods_per_year": 12,
        "skew": 0.0,
        "kurtosis": 3.0,
        "n_trials": 100,
        "var_trials_annual": 0.5,
        "n_tests": 100,
        "autocorr": 0.1,
    },
    "strong": {
        "sr_annual": 5.0,
        "n_obs": 240,
        "periods_per_year": 12,
        "skew": 0.0,
        "kurtosis": 3.0,
        "n_trials": 100,
        "var_trials_annual": 0.5,
        "n_tests": 1000,
        "autocorr": 0.0,
    },
}


def scenarios() -> dict[str, dict[str, Any]]:
    """Return the named scenarios and a seeded sample of the grid."""
    rng = random.Random(SEED)
    found = dict(NAMED)
    for k in range(N_SCENARIOS):
        inputs: dict[str, Any] = {name: rng.choice(values) for name, values in GRID.items()}
        if k % 2:
            inputs["n_obs"] = round(inputs.pop("years") * inputs["periods_per_year"])
        found[f"grid_{k:03d}"] = inputs
    return found


def outputs(inputs: dict[str, Any]) -> dict[str, float]:
    """Return what the page shows for ``inputs``, leaving out any the library refuses."""
    q = inputs["periods_per_year"]
    n_obs = inputs["n_obs"] if "n_obs" in inputs else round(inputs["years"] * q)
    sr = inputs["sr_annual"] / math.sqrt(q)
    skew, kurtosis = inputs["skew"], inputs["kurtosis"]
    n_trials = inputs["n_trials"]
    variance = inputs["var_trials_annual"] / q
    found: dict[str, float] = {}

    sr0 = ol.expected_max_sharpe(n_trials, variance)
    found["sr0Annual"] = ol.annualize_sharpe(sr0, q)
    try:
        found["psr0"] = ol.probabilistic_sharpe_ratio(sr, n_obs, skew=skew, kurtosis=kurtosis)
        found["dsr"] = ol.deflated_sharpe_ratio(
            sr, n_obs, n_trials=n_trials, var_trials=variance, skew=skew, kurtosis=kurtosis
        ).dsr
    except ValueError:
        pass
    try:
        length = ol.min_track_record_length(sr, skew=skew, kurtosis=kurtosis)
        found["minTrlObs"] = length
        found["minTrlYears"] = length / q
    except ValueError:
        pass
    if inputs["sr_annual"] > 0.0:
        found["minBtlYears"] = ol.min_backtest_length(
            n_trials, max_sharpe_annual=inputs["sr_annual"]
        )
    try:
        haircut = ol.haircut_sharpe(
            inputs["sr_annual"],
            n_obs,
            periods_per_year=q,
            n_tests=inputs["n_tests"],
            method="bonferroni",
            autocorr=inputs["autocorr"],
        )
        found["haircutSrAnnual"] = haircut.haircut_sr_annual
        found["haircutPct"] = haircut.haircut_pct
    except ValueError:
        pass
    return found


def numeric_cases() -> list[dict[str, Any]]:
    """Return 200 points each of the normal CDF and PPF, and a set for Student's t."""
    xs = np.linspace(-8.0, 8.0, 200)
    tail = np.logspace(-12.0, -3.0, 40)
    centre = np.linspace(0.001, 0.999, 120)
    ps = np.concatenate([tail, centre, 1.0 - np.logspace(-10.0, -3.0, 40)])
    cases: list[dict[str, Any]] = []
    for k, x in enumerate(xs):
        cases.append(
            {"id": f"cdf_{k:03d}", "fn": "normCdf", "inputs": {"x": x}, "expected": ndtr(x)}
        )
    for k, p in enumerate(ps):
        cases.append(
            {"id": f"ppf_{k:03d}", "fn": "normPpf", "inputs": {"p": p}, "expected": ndtri(p)}
        )
    for df in (1, 2, 5, 11, 59, 119, 239, 1199):
        for t in (-30.0, -4.0, -1.0, 0.0, 0.5, 2.0, 6.0):
            cases.append(
                {
                    "id": f"tcdf_{df}_{t}",
                    "fn": "studentCdf",
                    "inputs": {"df": df, "t": t},
                    "expected": stdtr(df, t),
                }
            )
        for p in (1e-8, 0.001, 0.025, 0.2, 0.5, 0.9, 0.995):
            cases.append(
                {
                    "id": f"tppf_{df}_{p}",
                    "fn": "studentPpf",
                    "inputs": {"df": df, "p": p},
                    "expected": stdtrit(df, p),
                }
            )
    return cases


def build_cases() -> list[dict[str, Any]]:
    """Return every case, numeric first."""
    cases = numeric_cases()
    for name, inputs in scenarios().items():
        for fn, expected in outputs(inputs).items():
            cases.append({"id": f"{name}_{fn}", "fn": fn, "inputs": inputs, "expected": expected})
    return [
        {
            **case,
            "inputs": {key: float(v) for key, v in case["inputs"].items()},
            "expected": float(case["expected"]),
        }
        for case in cases
    ]


def main() -> None:
    """Write the fixture, one case per line."""
    lines = [json.dumps(case, separators=(",", ":")) for case in build_cases()]
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("[\n" + ",\n".join(lines) + "\n]\n")


if __name__ == "__main__":
    main()
