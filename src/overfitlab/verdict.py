"""Verdict gate: combine the evidence into PASS, FAIL or INCONCLUSIVE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class VerdictConfig:
    """Thresholds of the verdict gate, with the defaults of decision D-06.

    Attributes
    ----------
    dsr_min : float
        Minimum Deflated Sharpe Ratio.
    pbo_max : float
        Maximum Probability of Backtest Overfitting.
    snoop_alpha : float
        Significance level of the SPA and stepdown tests.
    haircut_method : {"bonferroni", "holm", "bhy"}
        Multiple-testing adjustment of the haircut.
    pbo_partitions : int
        Number of CSCV partitions ``S``.
    n_boot : int
        Number of bootstrap resamples.
    periods_per_year : int
        Periods per year of the returns.
    """

    dsr_min: float = 0.95
    pbo_max: float = 0.20
    snoop_alpha: float = 0.05
    haircut_method: Literal["bonferroni", "holm", "bhy"] = "bhy"
    pbo_partitions: int = 16
    n_boot: int = 1000
    periods_per_year: int = 252
