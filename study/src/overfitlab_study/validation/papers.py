"""Golden runner: evaluate tests/golden/paper_examples.yaml against the library."""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

import overfitlab as ol
from overfitlab._normal import norm_sf, upper_quantile
from overfitlab.haircut import Method

Inputs = Mapping[str, Any]

COLUMNS = ("id", "source", "printed", "computed", "abs_diff", "tolerance", "pass", "notes")


def _dsr(a: Inputs) -> dict[str, float]:
    q = a["periods_per_year"]
    result = ol.deflated_sharpe_ratio(
        a["sr_annual"] / math.sqrt(q),
        a["n_obs"],
        n_trials=a["n_trials"],
        var_trials=a["var_trials_annual"] / q,
        skew=a["skew"],
        kurtosis=a["kurtosis"],
    )
    return {"sr0": result.sr0, "dsr": result.dsr}


def _mintrl_obs(a: Inputs) -> float:
    root = math.sqrt(a["periods_per_year"])
    return ol.min_track_record_length(
        a["sr_annual"] / root,
        sr_benchmark=a["benchmark_annual"] / root,
        skew=a.get("skew", 0.0),
        kurtosis=a.get("kurtosis", 3.0),
    )


def _haircut(a: Inputs) -> dict[str, float]:
    result = ol.haircut_sharpe(
        a["sr_annual"],
        a["n_obs"],
        periods_per_year=a["periods_per_year"],
        n_tests=a["n_tests"],
        method=cast(Method, a["method"]),
        autocorr=a["autocorr"],
        rho_tests=a["rho_tests"],
    )
    return {
        "sr_annual": result.sr_annual,
        "adj_pvalue": result.adj_pvalue,
        "haircut_sr_annual": result.haircut_sr_annual,
        "haircut_pct": result.haircut_pct,
    }


FUNCTIONS: dict[str, Callable[[Inputs], Mapping[str, float]]] = {
    "dsr": _dsr,
    "minbtl": lambda a: {
        "years": ol.min_backtest_length(a["n_trials"], max_sharpe_annual=a["max_sharpe_annual"])
    },
    "psr": lambda a: {
        "psr": ol.probabilistic_sharpe_ratio(
            a["sr"], a["n_obs"], skew=a["skew"], kurtosis=a["kurtosis"]
        )
    },
    "psr_annual": lambda a: {
        "psr": ol.probabilistic_sharpe_ratio(
            a["sr_annual"] / math.sqrt(a["periods_per_year"]),
            a["n_obs"],
            skew=a["skew"],
            kurtosis=a["kurtosis"],
        )
    },
    "mintrl_obs": lambda a: {"observations": _mintrl_obs(a)},
    "mintrl_years": lambda a: {"years": _mintrl_obs(a) / a["periods_per_year"]},
    "haircut": _haircut,
    "lo_scale": lambda a: {"eta": ol.annualize_sharpe(1.0, a["q"], autocorr=a["rho"])},
    "bonferroni_tstat": lambda a: {"tstat": upper_quantile(a["alpha"] / (2.0 * a["n_tests"]))},
    "tstat_pvalue_pct": lambda a: {"pvalue_pct": 200.0 * norm_sf(a["tstat"])},
}


@dataclass(frozen=True)
class CaseResult:
    id: str
    group: str
    source: str
    status: str
    printed: float | None
    computed: float | None
    tolerance: float
    notes: str
    known_discrepancy: bool = False

    @property
    def abs_diff(self) -> float | None:
        if self.printed is None or self.computed is None:
            return None
        return abs(self.printed - self.computed)

    @property
    def passed(self) -> bool:
        return self.abs_diff is not None and self.abs_diff <= self.tolerance


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cast(list[dict[str, Any]], data["cases"])


def evaluate_case(case: Mapping[str, Any]) -> CaseResult:
    computed: float | None = None
    if case["status"] == "sourced":
        computed = float(FUNCTIONS[case["fn"]](case["inputs"])[case["output"]])
    return CaseResult(
        id=case["id"],
        group=case["group"],
        source=case["source"],
        status=case["status"],
        printed=case["printed"],
        computed=computed,
        tolerance=case["tolerance"],
        notes=case.get("notes", ""),
        known_discrepancy=case.get("known_discrepancy", False),
    )


def write_report(results: list[CaseResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(COLUMNS)
        for r in results:
            writer.writerow(
                [
                    r.id,
                    r.source,
                    "" if r.printed is None else r.printed,
                    "" if r.computed is None else f"{r.computed:.6g}",
                    "" if r.abs_diff is None else f"{r.abs_diff:.6g}",
                    r.tolerance,
                    r.status if r.status != "sourced" else str(r.passed).lower(),
                    r.notes,
                ]
            )
