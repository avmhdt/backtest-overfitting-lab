"""Closed-form reference power, derived in docs/theory/power.md.

These formulas use only the normal distribution, and share nothing with the simulated tests
they are checked against.
"""

from __future__ import annotations

import math

from overfitlab._normal import EULER_GAMMA, norm_cdf, norm_ppf, upper_quantile

MAX_STAT_TESTS = ("dsr", "rc", "spa", "stepm")


def per_period_sharpe(true_sharpe_annual: float, periods_per_year: int) -> float:
    return true_sharpe_annual / math.sqrt(periods_per_year)


def expected_max_normal(n_trials: int) -> float:
    """E[max of N standard normals], by the approximation of FR-05."""
    if n_trials == 1:
        return 0.0
    return (1.0 - EULER_GAMMA) * upper_quantile(1.0 / n_trials) + EULER_GAMMA * upper_quantile(
        1.0 / (n_trials * math.e)
    )


def psr_power(sr: float, n_obs: int, alpha: float) -> float:
    """Power of PSR(0) with one trial: Phi(SR sqrt(T-1) / sqrt(1 + SR^2/2) - z)."""
    return norm_cdf(
        sr * math.sqrt(n_obs - 1) / math.sqrt(1.0 + sr * sr / 2.0) - norm_ppf(1 - alpha)
    )


def max_stat_power(sr: float, n_obs: int, n_trials: int, critical: float) -> float:
    """P(max of N scores exceeds c): one shifted by SR sqrt(T), the rest standard normal."""
    delta = sr * math.sqrt(n_obs)
    return 1.0 - norm_cdf(critical - delta) * norm_cdf(critical) ** (n_trials - 1)


def critical_value(test: str, n_trials: int, alpha: float) -> float:
    """c = Phi^-1((1 - alpha)^(1/N)) for the snooping tests, e_N + z for DSR."""
    if test == "dsr":
        return expected_max_normal(n_trials) + norm_ppf(1.0 - alpha)
    return norm_ppf((1.0 - alpha) ** (1.0 / n_trials))


def reference_power(
    test: str,
    *,
    true_sharpe_annual: float,
    years: float,
    n_trials: int,
    alpha: float,
    periods_per_year: int,
) -> float:
    """Reference power of ``test`` in a cell of the check."""
    sr = per_period_sharpe(true_sharpe_annual, periods_per_year)
    n_obs = round(years * periods_per_year)
    if test == "psr":
        return psr_power(sr, n_obs, alpha)
    if test in MAX_STAT_TESTS:
        return max_stat_power(sr, n_obs, n_trials, critical_value(test, n_trials, alpha))
    raise ValueError(f"no reference power for {test!r}")
