"""Harvey-Liu Sharpe ratio haircut.

Follows Harvey and Liu (2015), "Backtesting", and the conventions of their reference
code ``Haircut_SR.m``: t-statistics on monthly observations with a Student t distribution
for the strategy under test, Bonferroni over ``n_tests``, and Holm and BHY over
``n_tests + 1`` with the other tests simulated from the Harvey-Liu-Zhu model.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr, stdtr, stdtrit

from overfitlab.results import HaircutResult
from overfitlab.sharpe import ar1_scale

F64 = NDArray[np.float64]
RNGLike = np.random.Generator | int | None
Method = Literal["bonferroni", "holm", "bhy"]

# Rows of rho, p0 and lambda from the reference code. Exhibit 7 of the paper prints the
# same model to three decimals with small differences; see docs/validation.md.
_HLZ_RHO = (0.0, 0.2, 0.4, 0.6, 0.8)
_HLZ_P0 = (0.39660, 0.44589, 0.48604, 0.59902, 0.83901)
_HLZ_LAMBDA = (0.0054995, 0.0055508, 0.0055413, 0.0055512, 0.0055956)
_HLZ_VOL_MONTHLY = 0.15 / math.sqrt(12.0)
_HLZ_MONTHS = 240
_CHUNK = 256


def _monthly_obs(n_obs: int, periods_per_year: int) -> int:
    months = math.floor(n_obs * 12 / periods_per_year)
    if months < 2:
        raise ValueError("the sample must span at least two months")
    return months


def _annual_after_autocorr(sr_annual: float, periods_per_year: int, autocorr: float) -> float:
    if autocorr == 0.0 or periods_per_year == 1:
        return sr_annual
    return sr_annual * ar1_scale(autocorr, periods_per_year)


def sharpe_to_tstat(
    sr_annual: float, n_obs: int, periods_per_year: int, *, autocorr: float = 0.0
) -> float:
    r"""Convert an annual Sharpe ratio to the t-statistic of Harvey and Liu.

    :math:`t = SR_{ann}/\sqrt{12}\cdot\sqrt{M}`, with ``M`` the whole number of months in
    the sample and :math:`SR_{ann}` corrected for autocorrelation by Lo (2002).

    Parameters
    ----------
    sr_annual : float
        Annual Sharpe ratio.
    n_obs : int
        Number of observations at ``periods_per_year``.
    periods_per_year : int
        Observations per year.
    autocorr : float, default 0.0
        First-order autocorrelation at the sampling frequency.

    Returns
    -------
    float
        t-statistic.

    Raises
    ------
    ValueError
        If the sample spans fewer than two months or ``autocorr`` is out of range.

    Examples
    --------
    >>> round(sharpe_to_tstat(1.0, 120, 12), 4)
    3.1623
    """
    months = _monthly_obs(n_obs, periods_per_year)
    return (
        _annual_after_autocorr(sr_annual, periods_per_year, autocorr)
        / math.sqrt(12.0)
        * (math.sqrt(months))
    )


def _hlz_parameters(rho_tests: float) -> tuple[float, float]:
    """Interpolate p0 and lambda linearly; above 0.8 extrapolate from the last two rows."""
    if not 0.0 <= rho_tests < 1.0:
        raise ValueError("rho_tests must be in [0, 1)")
    upper = min(int(rho_tests / 0.2) + 1, len(_HLZ_RHO) - 1)
    lower = upper - 1
    weight = (rho_tests - _HLZ_RHO[lower]) / 0.2
    p0 = (1.0 - weight) * _HLZ_P0[lower] + weight * _HLZ_P0[upper]
    lam = (1.0 - weight) * _HLZ_LAMBDA[lower] + weight * _HLZ_LAMBDA[upper]
    return p0, lam


def _simulate_pvalues(n_sims: int, n_tests: int, rho_tests: float, rng: np.random.Generator) -> F64:
    """Draw two-sided p-values of the other tests, shape ``(n_sims, n_tests)``."""
    p0, lam = _hlz_parameters(rho_tests)
    scale = _HLZ_VOL_MONTHLY / math.sqrt(_HLZ_MONTHS)
    common = rng.standard_normal((n_sims, 1))
    idio = rng.standard_normal((n_sims, n_tests))
    shocks = (math.sqrt(rho_tests) * common + math.sqrt(1.0 - rho_tests) * idio) * scale
    is_true = rng.random((n_sims, n_tests)) > p0
    means = rng.exponential(lam, (n_sims, n_tests))
    tstats = np.abs(is_true * means + shocks) / scale
    return 2.0 * ndtr(-tstats)


def _adjusted_for_last(pvalues: F64, method: Method) -> F64:
    """Return, per row, the adjusted p-value of the last column among all columns."""
    n_rows, n_all = pvalues.shape
    order = np.argsort(pvalues, axis=1)
    ordered = np.take_along_axis(pvalues, order, axis=1)
    if method == "holm":
        multipliers = np.arange(n_all, 0, -1, dtype=np.float64)
        adjusted = np.minimum(np.maximum.accumulate(ordered * multipliers, axis=1), 1.0)
    else:
        harmonic = float(np.sum(1.0 / np.arange(1, n_all + 1)))
        multipliers = n_all * harmonic / np.arange(1, n_all + 1, dtype=np.float64)
        scaled = ordered * multipliers
        # The reference code leaves the largest p-value unscaled.
        scaled[:, -1] = ordered[:, -1]
        adjusted = np.minimum.accumulate(scaled[:, ::-1], axis=1)[:, ::-1]
    position = np.argmax(order == n_all - 1, axis=1)
    return adjusted[np.arange(n_rows), position]


def _simulated_adjusted_pvalue(
    pvalue: float, n_tests: int, method: Method, rho_tests: float, n_sims: int, rng: RNGLike
) -> float:
    generator = np.random.default_rng(rng)
    values: list[F64] = []
    for start in range(0, n_sims, _CHUNK):
        size = min(_CHUNK, n_sims - start)
        others = _simulate_pvalues(size, n_tests, rho_tests, generator)
        all_p = np.column_stack((others, np.full(size, pvalue)))
        values.append(_adjusted_for_last(all_p, method))
    return float(np.median(np.concatenate(values)))


def haircut_sharpe(
    sr_annual: float,
    n_obs: int,
    *,
    periods_per_year: int,
    n_tests: int,
    method: Method = "bhy",
    autocorr: float = 0.0,
    rho_tests: float = 0.2,
    n_sims: int = 5000,
    rng: RNGLike = 0,
) -> HaircutResult:
    r"""Return the Sharpe ratio left after a multiple-testing haircut.

    The annual Sharpe ratio is corrected for autocorrelation, converted to a t-statistic
    and a two-sided p-value on ``M - 1`` degrees of freedom, adjusted for multiple
    testing and inverted back to an annual Sharpe ratio, following Harvey and Liu (2015).
    Bonferroni is exact, ``min(n_tests * p, 1)``. Holm and BHY take the median adjusted
    p-value over ``n_sims`` draws of the other tests from the Harvey-Liu-Zhu model with
    pairwise correlation ``rho_tests``.

    A non-positive ``sr_annual`` has no evidence to deflate: the p-values are 1, the
    haircut Sharpe ratio is 0 and the haircut is 100%.

    Parameters
    ----------
    sr_annual : float
        Annual Sharpe ratio.
    n_obs : int
        Number of observations at ``periods_per_year``.
    periods_per_year : int
        Observations per year.
    n_tests : int
        Number of tests.
    method : {"bonferroni", "holm", "bhy"}, default "bhy"
        Multiple-testing adjustment.
    autocorr : float, default 0.0
        First-order autocorrelation at the sampling frequency.
    rho_tests : float, default 0.2
        Average correlation among strategy returns, in ``[0, 1)``.
    n_sims : int, default 5000
        Number of simulated sets of other tests.
    rng : numpy.random.Generator or int or None, default 0
        Source of randomness for Holm and BHY.

    Returns
    -------
    HaircutResult
        Corrected and haircut Sharpe ratios, t-statistic and p-values.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> r = haircut_sharpe(1.0, 120, periods_per_year=12, n_tests=100, method="bonferroni",
    ...                    autocorr=0.1, rho_tests=0.4)
    >>> round(r.haircut_sr_annual, 3), round(r.adj_pvalue, 3)
    (0.232, 0.465)
    """
    if n_tests < 1:
        raise ValueError("n_tests must be at least 1")
    if method not in ("bonferroni", "holm", "bhy"):
        raise ValueError(f"unknown method {method!r}")
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    months = _monthly_obs(n_obs, periods_per_year)
    corrected = _annual_after_autocorr(sr_annual, periods_per_year, autocorr)
    tstat = corrected / math.sqrt(12.0) * math.sqrt(months)
    if corrected <= 0.0:
        return HaircutResult(method, corrected, 0.0, 100.0, tstat, 1.0, 1.0)

    dof = months - 1
    pvalue = float(2.0 * stdtr(dof, -tstat))
    if method == "bonferroni":
        adjusted = min(n_tests * pvalue, 1.0)
    else:
        adjusted = _simulated_adjusted_pvalue(pvalue, n_tests, method, rho_tests, n_sims, rng)
    tail = min(tstat, abs(float(stdtrit(dof, adjusted / 2.0))))
    haircut_sr = tail / math.sqrt(months) * math.sqrt(12.0)
    haircut_pct = 100.0 * (corrected - haircut_sr) / corrected
    return HaircutResult(method, corrected, haircut_sr, haircut_pct, tstat, pvalue, adjusted)
