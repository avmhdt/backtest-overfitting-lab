"""Probability of Backtest Overfitting by combinatorially symmetric cross-validation.

Implements CSCV of Bailey, Borwein, Lopez de Prado and Zhu (2017). Rows are cut into ``S``
contiguous blocks. Each way of choosing ``S/2`` blocks is an in-sample set and its complement
is the out-of-sample set. PBO is how often the in-sample winner lands in the lower half of
the out-of-sample ranking.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numba
import numpy as np
from numpy.typing import ArrayLike, NDArray

from overfitlab._arrays import as_matrix

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
Metric = Literal["sharpe", "mean"] | Callable[[F64], F64]


@dataclass(frozen=True)
class PBOResult:
    """Outcome of CSCV.

    Attributes
    ----------
    pbo : float
        Share of combinations whose logit is at or below zero.
    logits : numpy.ndarray
        ``ln(w / (1 - w))`` per combination.
    oos_rank : numpy.ndarray
        Relative out-of-sample rank ``w`` of the in-sample winner.
    is_perf : numpy.ndarray
        In-sample performance of the winner.
    oos_perf : numpy.ndarray
        Its out-of-sample performance.
    degradation_slope : float
        Slope of out-of-sample on in-sample performance.
    degradation_intercept : float
        Intercept of that regression.
    prob_oos_loss : float
        Share of combinations in which the winner loses out of sample.
    n_combinations : int
        ``C(S, S/2)``.
    """

    pbo: float
    logits: F64
    oos_rank: F64
    is_perf: F64
    oos_perf: F64
    degradation_slope: float
    degradation_intercept: float
    prob_oos_loss: float
    n_combinations: int


@numba.njit(cache=True)  # type: ignore[no-untyped-call,untyped-decorator,unused-ignore]
def _block_sums(returns: F64, block_size: int, n_blocks: int) -> tuple[F64, F64]:
    """Sum and sum of squares of each strategy in each block, ``(S, N)``."""
    n_strategies = returns.shape[1]
    sums = np.zeros((n_blocks, n_strategies))
    squares = np.zeros((n_blocks, n_strategies))
    for block in range(n_blocks):
        for t in range(block * block_size, (block + 1) * block_size):
            for j in range(n_strategies):
                value = returns[t, j]
                sums[block, j] += value
                squares[block, j] += value * value
    return sums, squares


@numba.njit(cache=True)  # type: ignore[no-untyped-call,untyped-decorator,unused-ignore]
def _metric_values(
    sums: F64, squares: F64, blocks: I64, block_size: int, use_sharpe: bool, out: F64
) -> None:
    n_strategies = sums.shape[1]
    n_obs = blocks.size * block_size
    for j in range(n_strategies):
        total = 0.0
        total_squares = 0.0
        for block in blocks:
            total += sums[block, j]
            total_squares += squares[block, j]
        mean = total / n_obs
        if use_sharpe:
            variance = (total_squares - n_obs * mean * mean) / (n_obs - 1)
            out[j] = mean / np.sqrt(variance) if variance > 1e-30 else 0.0
        else:
            out[j] = mean


@numba.njit(cache=True)  # type: ignore[no-untyped-call,untyped-decorator,unused-ignore]
def _winner_rank(values: F64, winner: int) -> float:
    """Average-tie rank of the winner: 1 is the worst and N the best."""
    below = 0
    equal = 0
    chosen = values[winner]
    for value in values:
        if value < chosen:
            below += 1
        elif value == chosen:
            equal += 1
    return below + (equal + 1) / 2.0


@numba.njit(cache=True)  # type: ignore[no-untyped-call,untyped-decorator,unused-ignore]
def _cscv(
    sums: F64, squares: F64, in_sample: I64, out_of_sample: I64, block_size: int, use_sharpe: bool
) -> tuple[F64, F64, F64]:
    n_combinations = in_sample.shape[0]
    n_strategies = sums.shape[1]
    ranks = np.zeros(n_combinations)
    is_perf = np.zeros(n_combinations)
    oos_perf = np.zeros(n_combinations)
    is_values = np.zeros(n_strategies)
    oos_values = np.zeros(n_strategies)
    for c in range(n_combinations):
        _metric_values(sums, squares, in_sample[c], block_size, use_sharpe, is_values)
        _metric_values(sums, squares, out_of_sample[c], block_size, use_sharpe, oos_values)
        winner = int(np.argmax(is_values))
        ranks[c] = _winner_rank(oos_values, winner)
        is_perf[c] = is_values[winner]
        oos_perf[c] = oos_values[winner]
    return ranks, is_perf, oos_perf


def _callable_cscv(
    returns: F64,
    in_sample: I64,
    out_of_sample: I64,
    block_size: int,
    metric: Callable[[F64], F64],
) -> tuple[F64, F64, F64]:
    n_combinations = in_sample.shape[0]
    ranks = np.zeros(n_combinations)
    is_perf = np.zeros(n_combinations)
    oos_perf = np.zeros(n_combinations)

    def rows(blocks: I64) -> I64:
        return np.concatenate([np.arange(b * block_size, (b + 1) * block_size) for b in blocks])

    for c in range(n_combinations):
        is_values = np.asarray(metric(returns[rows(in_sample[c])]), dtype=np.float64)
        oos_values = np.asarray(metric(returns[rows(out_of_sample[c])]), dtype=np.float64)
        winner = int(np.argmax(is_values))
        ranks[c] = _winner_rank(oos_values, winner)
        is_perf[c] = is_values[winner]
        oos_perf[c] = oos_values[winner]
    return ranks, is_perf, oos_perf


def pbo_cscv(
    returns: ArrayLike,
    *,
    n_partitions: int = 16,
    metric: Metric = "sharpe",
) -> PBOResult:
    r"""Return the Probability of Backtest Overfitting.

    For each of the :math:`C(S, S/2)` combinations of blocks, the trial with the best
    in-sample metric has a relative out-of-sample rank :math:`\omega = r / (N + 1)` and a
    logit :math:`\lambda = \ln(\omega / (1 - \omega))`. PBO is the share of
    :math:`\lambda \le 0`, from Bailey et al. (2017). Rank ties take the average rank. Rows
    beyond the last whole block are dropped, as in the paper.

    Parameters
    ----------
    returns : array_like
        ``T x N`` matrix of per-period returns.
    n_partitions : int, default 16
        Number of blocks ``S``, even.
    metric : {"sharpe", "mean"} or callable, default "sharpe"
        A callable maps a ``T' x N`` matrix to ``N`` values, larger meaning better.

    Returns
    -------
    PBOResult
        PBO, logits, performance degradation and probability of loss.

    Raises
    ------
    ValueError
        If ``n_partitions`` is odd or too small, there are fewer than two trials, there are
        too few rows, or ``metric`` is unknown.

    Examples
    --------
    >>> r = np.random.default_rng(0).normal(0.0, 0.01, (320, 50))
    >>> result = pbo_cscv(r, n_partitions=8)
    >>> result.n_combinations
    70
    >>> 0.0 <= result.pbo <= 1.0
    True
    """
    matrix = as_matrix(returns)
    n_obs, n_strategies = matrix.shape
    if n_partitions < 2 or n_partitions % 2 != 0:
        raise ValueError("n_partitions must be even and at least 2")
    if n_strategies < 2:
        raise ValueError("at least two trials are needed")
    block_size = n_obs // n_partitions
    if block_size < 1 or n_partitions * block_size < 2:
        raise ValueError("too few observations for n_partitions")
    if isinstance(metric, str) and metric not in ("sharpe", "mean"):
        raise ValueError(f"unknown metric {metric!r}")

    half = n_partitions // 2
    chosen = np.array(list(combinations(range(n_partitions), half)), dtype=np.int64)
    every = np.arange(n_partitions, dtype=np.int64)
    rest = np.array([np.setdiff1d(every, row) for row in chosen], dtype=np.int64)

    if callable(metric):
        ranks, is_perf, oos_perf = _callable_cscv(matrix, chosen, rest, block_size, metric)
    else:
        sums, squares = _block_sums(matrix, block_size, n_partitions)
        ranks, is_perf, oos_perf = _cscv(
            sums, squares, chosen, rest, block_size, metric == "sharpe"
        )

    oos_rank = ranks / (n_strategies + 1)
    logits = np.log(oos_rank / (1.0 - oos_rank))
    slope, intercept = 0.0, float(oos_perf.mean())
    spread = is_perf - is_perf.mean()
    if float(np.dot(spread, spread)) > 0.0:
        slope = float(np.dot(spread, oos_perf - oos_perf.mean()) / np.dot(spread, spread))
        intercept = float(oos_perf.mean() - slope * is_perf.mean())
    return PBOResult(
        pbo=float(np.mean(logits <= 0.0)),
        logits=logits,
        oos_rank=oos_rank,
        is_perf=is_perf,
        oos_perf=oos_perf,
        degradation_slope=slope,
        degradation_intercept=intercept,
        prob_oos_loss=float(np.mean(oos_perf < 0.0)),
        n_combinations=math.comb(n_partitions, half),
    )
