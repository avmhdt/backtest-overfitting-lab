"""Data-snooping tests: White's Reality Check, Hansen's SPA and the Romano-Wolf stepdown.

``rel_perf`` is a ``T x N`` matrix of per-period performance relative to the benchmark, with
larger meaning better. Bootstrap indices are drawn in NumPy and passed to a Numba kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numba
import numpy as np
from numpy.typing import ArrayLike, NDArray

from overfitlab._arrays import as_matrix
from overfitlab.bootstrap import optimal_block_length, stationary_bootstrap_indices

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
RNGLike = np.random.Generator | int | None


@dataclass(frozen=True)
class SnoopResult:
    """Reality Check outcome.

    Attributes
    ----------
    stat : float
        ``max_k sqrt(T) * mean(d_k)``.
    pvalue : float
        Bootstrap p-value.
    n_boot : int
        Resamples.
    block_length : float
        Mean block length, NaN when indices were supplied without one.
    """

    stat: float
    pvalue: float
    n_boot: int
    block_length: float


@dataclass(frozen=True)
class SPAResult:
    """Superior Predictive Ability outcome.

    Attributes
    ----------
    stat : float
        Studentised statistic, at least 0.
    pvalue_lower : float
        Lower-bound p-value.
    pvalue_consistent : float
        Consistent p-value.
    pvalue_upper : float
        Upper-bound p-value.
    n_boot : int
        Resamples.
    block_length : float
        Mean block length, NaN when indices were supplied without one.
    """

    stat: float
    pvalue_lower: float
    pvalue_consistent: float
    pvalue_upper: float
    n_boot: int
    block_length: float


@dataclass(frozen=True)
class StepMResult:
    """Stepdown outcome.

    Attributes
    ----------
    rejected : numpy.ndarray
        ``N`` flags, True for strategies better than the benchmark.
    adjusted_pvalues : numpy.ndarray
        Adjusted p-value of each strategy.
    n_steps : int
        Steps taken, including the last one that rejected nothing.
    alpha : float
        Familywise error rate.
    """

    rejected: NDArray[np.bool_]
    adjusted_pvalues: F64
    n_steps: int
    alpha: float


@numba.njit(cache=True)  # type: ignore[no-untyped-call,untyped-decorator,unused-ignore]
def _bootstrap_means(values: F64, indices: I64) -> F64:
    """Column means of each resample, shape ``(n_boot, N)``."""
    n_boot, n_obs = indices.shape
    n_strategies = values.shape[1]
    out = np.zeros((n_boot, n_strategies))
    for b in range(n_boot):
        for t in range(n_obs):
            row = indices[b, t]
            for j in range(n_strategies):
                out[b, j] += values[row, j]
    return out / n_obs  # type: ignore[no-any-return,unused-ignore]


def _resample_plan(
    rel_perf: F64,
    n_boot: int,
    block_length: float | None,
    indices: I64 | None,
    rng: RNGLike,
) -> tuple[I64, float]:
    n_obs = rel_perf.shape[0]
    if indices is not None:
        plan = np.asarray(indices, dtype=np.int64)
        if plan.ndim != 2 or plan.shape[1] != n_obs:
            raise ValueError("indices must have shape (n_boot, T)")
        return plan, float("nan") if block_length is None else block_length
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if block_length is None:
        block_length = float(
            np.median([optimal_block_length(rel_perf[:, j]) for j in range(rel_perf.shape[1])])
        )
    plan = stationary_bootstrap_indices(n_obs, n_boot, mean_block=block_length, rng=rng)
    return plan, block_length


def _long_run_variance(rel_perf: F64, block_length: float) -> F64:
    """Hansen's (2005) estimator with the kernel of the stationary bootstrap.

    The block length is the bootstrap's. Autocovariances of every column come from one FFT,
    and a column with no variance gets an infinite scale, so its statistic is zero.
    """
    n_obs = rel_perf.shape[0]
    demeaned = rel_perf - rel_perf.mean(axis=0)
    transform = np.fft.rfft(demeaned, n=2 * n_obs, axis=0)
    autocov = np.fft.irfft(np.abs(transform) ** 2, axis=0)[:n_obs] / n_obs
    q = 1.0 / block_length
    lags = np.arange(n_obs, dtype=np.float64)
    weights = (1.0 - lags / n_obs) * (1.0 - q) ** lags + (lags / n_obs) * (1.0 - q) ** (
        n_obs - lags
    )
    weights[0] = 0.5
    variance = 2.0 * (weights[:, np.newaxis] * autocov).sum(axis=0)
    fallback = autocov[0]
    variance = np.where(variance > 0.0, variance, fallback)
    return np.asarray(np.where(variance > 0.0, variance, np.inf), dtype=np.float64)


def reality_check(
    rel_perf: ArrayLike,
    *,
    n_boot: int = 1000,
    block_length: float | None = None,
    indices: I64 | None = None,
    rng: RNGLike = None,
) -> SnoopResult:
    r"""Return White's (2000) Reality Check.

    The statistic is :math:`\max_k \sqrt{T}\,\bar{d}_k`, and the p-value is the share of
    resamples whose recentred maximum :math:`\max_k \sqrt{T}(\bar{d}^*_k - \bar{d}_k)`
    reaches it.

    Parameters
    ----------
    rel_perf : array_like
        ``T x N`` per-period performance relative to the benchmark, larger meaning better.
    n_boot : int, default 1000
        Resamples.
    block_length : float, optional
        Mean block length. Defaults to the median of Politis and White over the columns.
    indices : numpy.ndarray, optional
        Bootstrap indices of shape ``(n_boot, T)``, which replace ``n_boot`` and ``rng``.
    rng : numpy.random.Generator or int or None, optional
        Source of randomness.

    Returns
    -------
    SnoopResult
        Statistic and p-value.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> d = np.random.default_rng(0).normal(0.0, 0.01, (250, 5))
    >>> 0.0 <= reality_check(d, n_boot=200, block_length=2.0, rng=0).pvalue <= 1.0
    True
    """
    matrix = as_matrix(rel_perf, name="rel_perf")
    plan, block = _resample_plan(matrix, n_boot, block_length, indices, rng)
    root = math.sqrt(matrix.shape[0])
    means = matrix.mean(axis=0)
    resampled = _bootstrap_means(matrix, plan)
    maxima = root * (resampled - means).max(axis=1)
    stat = root * float(means.max())
    return SnoopResult(stat, float(np.mean(maxima >= stat)), plan.shape[0], block)


def spa_test(
    rel_perf: ArrayLike,
    *,
    n_boot: int = 1000,
    block_length: float | None = None,
    indices: I64 | None = None,
    rng: RNGLike = None,
    studentize: bool = True,
) -> SPAResult:
    r"""Return Hansen's (2005) test for Superior Predictive Ability.

    The statistic is :math:`\max(\max_k \sqrt{T}\,\bar{d}_k / \hat{\omega}_k, 0)`, with
    :math:`\hat{\omega}_k^2` the long-run variance under the stationary bootstrap's kernel.
    Resamples are recentred by :math:`\max(\bar{d}_k, 0)` for the lower p-value, by
    :math:`\bar{d}_k` for the upper one, and for the consistent one by :math:`\bar{d}_k`
    unless it falls below :math:`-\hat{\omega}_k\sqrt{2\ln\ln T / T}`, when it is zero.

    Parameters
    ----------
    rel_perf : array_like
        ``T x N`` per-period performance relative to the benchmark, larger meaning better.
    n_boot : int, default 1000
        Resamples.
    block_length : float, optional
        Mean block length. Defaults to the median of Politis and White over the columns.
    indices : numpy.ndarray, optional
        Bootstrap indices of shape ``(n_boot, T)``. Then ``block_length`` should be the mean
        block used for them, or the variance kernel assumes the default.
    rng : numpy.random.Generator or int or None, optional
        Source of randomness.
    studentize : bool, default True
        Hansen's statistic. False compares raw means with no floor at zero, which is the
        definition of the ``arch`` package, used by the cross-check.

    Returns
    -------
    SPAResult
        Statistic and the three p-values.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> d = np.random.default_rng(0).normal(0.0, 0.01, (250, 5))
    >>> result = spa_test(d, n_boot=200, block_length=2.0, rng=0)
    >>> result.pvalue_lower <= result.pvalue_consistent <= result.pvalue_upper
    True
    """
    matrix = as_matrix(rel_perf, name="rel_perf")
    plan, block = _resample_plan(matrix, n_boot, block_length, indices, rng)
    n_obs = matrix.shape[0]
    root = math.sqrt(n_obs)
    kernel_block = block if math.isfinite(block) else math.sqrt(n_obs)
    omega = np.sqrt(_long_run_variance(matrix, kernel_block))
    means = matrix.mean(axis=0)
    resampled = _bootstrap_means(matrix, plan)

    threshold = -omega * math.sqrt(2.0 * math.log(math.log(n_obs)) / n_obs)
    centres = {
        "lower": np.maximum(means, 0.0),
        "consistent": np.where(means >= threshold, means, 0.0),
        "upper": means,
    }
    scale = omega if studentize else 1.0
    floor = 0.0 if studentize else -np.inf
    stat = max(float((root * means / scale).max()), floor)
    pvalues = {}
    for name, centre in centres.items():
        simulated = np.maximum((root * (resampled - centre) / scale).max(axis=1), floor)
        pvalues[name] = float(np.mean(simulated > stat))
    return SPAResult(
        stat,
        pvalues["lower"],
        pvalues["consistent"],
        pvalues["upper"],
        plan.shape[0],
        block,
    )


def romano_wolf_stepdown(
    rel_perf: ArrayLike,
    *,
    alpha: float = 0.05,
    n_boot: int = 1000,
    block_length: float | None = None,
    indices: I64 | None = None,
    rng: RNGLike = None,
) -> StepMResult:
    r"""Return the studentised stepdown of Romano and Wolf (2005).

    Strategies are sorted by :math:`t_k = \sqrt{T}\,\bar{d}_k / \hat{\omega}_k`. Each step
    rejects those whose :math:`t_k` beats the bootstrap distribution of the maximum over the
    strategies not yet rejected, and repeats until a step rejects nothing, which controls the
    familywise error rate at ``alpha``. Adjusted p-values follow Romano and Wolf (2016).

    Parameters
    ----------
    rel_perf : array_like
        ``T x N`` per-period performance relative to the benchmark, larger meaning better.
    alpha : float, default 0.05
        Familywise error rate.
    n_boot : int, default 1000
        Resamples.
    block_length : float, optional
        Mean block length. Defaults to the median of Politis and White over the columns.
    indices : numpy.ndarray, optional
        Bootstrap indices of shape ``(n_boot, T)``.
    rng : numpy.random.Generator or int or None, optional
        Source of randomness.

    Returns
    -------
    StepMResult
        Rejected set and adjusted p-values.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> d = np.random.default_rng(0).normal(0.0, 0.01, (500, 5))
    >>> d[:, 0] += 0.01
    >>> romano_wolf_stepdown(d, n_boot=200, block_length=2.0, rng=0).rejected.tolist()
    [True, False, False, False, False]
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    matrix = as_matrix(rel_perf, name="rel_perf")
    plan, block = _resample_plan(matrix, n_boot, block_length, indices, rng)
    n_obs = matrix.shape[0]
    root = math.sqrt(n_obs)
    kernel_block = block if math.isfinite(block) else math.sqrt(n_obs)
    omega = np.sqrt(_long_run_variance(matrix, kernel_block))
    means = matrix.mean(axis=0)
    tstats = root * means / omega
    simulated = root * (_bootstrap_means(matrix, plan) - means) / omega

    order = np.argsort(-tstats, kind="stable")
    sorted_t = tstats[order]
    remaining_max = np.maximum.accumulate(simulated[:, order[::-1]], axis=1)[:, ::-1]
    raw = (remaining_max >= sorted_t).mean(axis=0)
    adjusted_sorted = np.maximum.accumulate(raw)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted

    rejected = adjusted <= alpha
    n_steps = 1
    active = np.ones(tstats.size, dtype=bool)
    while active.any():
        maxima = simulated[:, active].max(axis=1)
        new = active & ((maxima[:, np.newaxis] >= tstats).mean(axis=0) <= alpha)
        if not new.any():
            break
        active &= ~new
        if active.any():
            n_steps += 1
    return StepMResult(rejected, adjusted, n_steps, alpha)
