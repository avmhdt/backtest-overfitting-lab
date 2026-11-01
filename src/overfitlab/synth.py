"""Synthetic data with known ground truth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from overfitlab._arrays import as_vector
from overfitlab.bootstrap import optimal_block_length, stationary_bootstrap_indices
from overfitlab.factory import StrategySpec, run_strategies
from overfitlab.sharpe import annualize_sharpe, sharpe_ratio

F64 = NDArray[np.float64]
RNGLike = np.random.Generator | int | None

_TREND_ORACLE = StrategySpec("ma_cross", {"fast": 10, "slow": 60, "mode": "long_short"})
_REVERSION_ORACLE = StrategySpec(
    "mean_reversion", {"lookback": 20, "entry_z": 1.0, "exit_z": 0.0, "mode": "long_short"}
)
_TREND_PHI = 0.98
_REVERSION_PHI = 0.90
_CALIBRATION_BARS = 100_000
_CALIBRATION_SEED = 20261030


@dataclass(frozen=True)
class SynthResult:
    """Return matrix with its ground truth.

    Attributes
    ----------
    returns : numpy.ndarray
        ``T x N`` per-period returns.
    is_true : numpy.ndarray
        ``N`` flags, True for strategies with a planted signal.
    true_sharpe_annual : numpy.ndarray
        True annual Sharpe ratio of each strategy.
    """

    returns: F64
    is_true: NDArray[np.bool_]
    true_sharpe_annual: F64


def _check_size(n_obs: int, n_series: int = 1) -> None:
    if n_obs < 1 or n_series < 1:
        raise ValueError("n_obs and n_series must be at least 1")


def gaussian_noise(
    n_obs: int,
    n_series: int = 1,
    *,
    vol_annual: float = 0.16,
    periods_per_year: int = 252,
    rng: RNGLike = None,
) -> F64:
    """Return zero-mean Gaussian returns.

    Parameters
    ----------
    n_obs : int
        Observations ``T``.
    n_series : int, default 1
        Series ``N``.
    vol_annual : float, default 0.16
        Annual volatility.
    periods_per_year : int, default 252
        Periods per year.
    rng : numpy.random.Generator or int or None, optional
        Source of randomness.

    Returns
    -------
    numpy.ndarray
        ``T x N`` per-period returns.

    Raises
    ------
    ValueError
        If a size is below 1.

    Examples
    --------
    >>> gaussian_noise(100, 3, rng=0).shape
    (100, 3)
    """
    _check_size(n_obs, n_series)
    generator = np.random.default_rng(rng)
    return generator.standard_normal((n_obs, n_series)) * vol_annual / math.sqrt(periods_per_year)


def student_t_noise(
    n_obs: int,
    n_series: int = 1,
    *,
    df: float = 4.0,
    vol_annual: float = 0.16,
    periods_per_year: int = 252,
    rng: RNGLike = None,
) -> F64:
    """Return zero-mean Student t returns, scaled to the chosen volatility.

    Parameters
    ----------
    n_obs : int
        Observations ``T``.
    n_series : int, default 1
        Series ``N``.
    df : float, default 4.0
        Degrees of freedom, above 2 so that the variance exists.
    vol_annual : float, default 0.16
        Annual volatility.
    periods_per_year : int, default 252
        Periods per year.
    rng : numpy.random.Generator or int or None, optional
        Source of randomness.

    Returns
    -------
    numpy.ndarray
        ``T x N`` per-period returns.

    Raises
    ------
    ValueError
        If ``df <= 2`` or a size is below 1.

    Examples
    --------
    >>> student_t_noise(100, 2, rng=0).shape
    (100, 2)
    """
    _check_size(n_obs, n_series)
    if df <= 2.0:
        raise ValueError("df must exceed 2")
    generator = np.random.default_rng(rng)
    draws = generator.standard_t(df, (n_obs, n_series)) / math.sqrt(df / (df - 2.0))
    return draws * vol_annual / math.sqrt(periods_per_year)


def gbm_prices(
    n_obs: int,
    *,
    mu_annual: float = 0.0,
    vol_annual: float = 0.16,
    s0: float = 100.0,
    periods_per_year: int = 252,
    rng: RNGLike = None,
) -> F64:
    """Return a geometric Brownian motion price path.

    Parameters
    ----------
    n_obs : int
        Number of prices, starting with ``s0``.
    mu_annual : float, default 0.0
        Annual drift.
    vol_annual : float, default 0.16
        Annual volatility.
    s0 : float, default 100.0
        First price.
    periods_per_year : int, default 252
        Periods per year.
    rng : numpy.random.Generator or int or None, optional
        Source of randomness.

    Returns
    -------
    numpy.ndarray
        ``n_obs`` prices.

    Raises
    ------
    ValueError
        If ``n_obs`` is below 1 or ``s0`` is not positive.

    Examples
    --------
    >>> float(gbm_prices(4, rng=0)[0])
    100.0
    """
    _check_size(n_obs)
    if s0 <= 0.0:
        raise ValueError("s0 must be positive")
    generator = np.random.default_rng(rng)
    dt = 1.0 / periods_per_year
    drift = (mu_annual - 0.5 * vol_annual**2) * dt
    steps = drift + vol_annual * math.sqrt(dt) * generator.standard_normal(n_obs - 1)
    return s0 * np.exp(np.concatenate(([0.0], np.cumsum(steps))))


def planted_signal_matrix(
    n_obs: int,
    n_strategies: int,
    *,
    true_sharpe_annual: float,
    n_true: int = 1,
    corr: float = 0.0,
    vol_annual: float = 0.16,
    periods_per_year: int = 252,
    rng: RNGLike = None,
) -> SynthResult:
    """Return a strategy matrix in which the first ``n_true`` columns have an edge.

    Parameters
    ----------
    n_obs : int
        Observations ``T``.
    n_strategies : int
        Strategies ``N``.
    true_sharpe_annual : float
        True annual Sharpe ratio of the planted strategies.
    n_true : int, default 1
        Planted strategies.
    corr : float, default 0.0
        Pairwise correlation of the shocks, in ``[0, 1)``.
    vol_annual : float, default 0.16
        Annual volatility.
    periods_per_year : int, default 252
        Periods per year.
    rng : numpy.random.Generator or int or None, optional
        Source of randomness.

    Returns
    -------
    SynthResult
        Returns, truth flags and true annual Sharpe ratios.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> synth = planted_signal_matrix(100, 5, true_sharpe_annual=1.0, n_true=2, rng=0)
    >>> synth.is_true.tolist()
    [True, True, False, False, False]
    """
    _check_size(n_obs, n_strategies)
    if not 0 <= n_true <= n_strategies:
        raise ValueError("n_true must be between 0 and n_strategies")
    if not 0.0 <= corr < 1.0:
        raise ValueError("corr must be in [0, 1)")
    generator = np.random.default_rng(rng)
    vol = vol_annual / math.sqrt(periods_per_year)
    common = generator.standard_normal((n_obs, 1))
    idio = generator.standard_normal((n_obs, n_strategies))
    returns = (math.sqrt(corr) * common + math.sqrt(1.0 - corr) * idio) * vol
    is_true = np.zeros(n_strategies, dtype=bool)
    is_true[:n_true] = True
    true_sharpe = np.where(is_true, true_sharpe_annual, 0.0)
    returns += true_sharpe / math.sqrt(periods_per_year) * vol
    return SynthResult(returns, is_true, true_sharpe)


def _ar1(n_obs: int, phi: float, generator: np.random.Generator) -> F64:
    """Return a stationary AR(1) path with unit variance."""
    shocks = generator.standard_normal(n_obs) * math.sqrt(1.0 - phi * phi)
    path = np.zeros(n_obs)
    path[0] = generator.standard_normal()
    for t in range(1, n_obs):
        path[t] = phi * path[t - 1] + shocks[t]
    return path


def _signal_prices(
    n_obs: int,
    strength: float,
    kind: str,
    vol: float,
    generator: np.random.Generator,
) -> F64:
    noise = generator.standard_normal(n_obs - 1) * vol
    if kind == "trend":
        component = _ar1(n_obs - 1, _TREND_PHI, generator)
        log_returns = strength * vol * component + noise
        log_path = np.concatenate(([0.0], np.cumsum(log_returns)))
    else:
        deviation = _ar1(n_obs, _REVERSION_PHI, generator)
        log_path = np.concatenate(([0.0], np.cumsum(noise))) + strength * vol * deviation
    return 100.0 * np.exp(log_path)


def _oracle_sharpe_annual(strength: float, kind: str, periods_per_year: int, vol: float) -> float:
    generator = np.random.default_rng(_CALIBRATION_SEED)
    prices = _signal_prices(_CALIBRATION_BARS, strength, kind, vol, generator)
    oracle = _TREND_ORACLE if kind == "trend" else _REVERSION_ORACLE
    returns = run_strategies(prices, [oracle], cost_bps=0.0)[:, 0]
    return annualize_sharpe(float(sharpe_ratio(returns)), periods_per_year)


@lru_cache(maxsize=64)
def _calibrate_strength(true_sharpe_annual: float, kind: str, periods_per_year: int) -> float:
    """Bisect the signal amplitude so the oracle earns the target Sharpe ratio before costs."""
    vol = 0.16 / math.sqrt(periods_per_year)
    low, high = 0.0, 1.0
    while _oracle_sharpe_annual(high, kind, periods_per_year, vol) < true_sharpe_annual:
        high *= 2.0
        if high > 1e4:
            raise ValueError(f"cannot reach a true Sharpe ratio of {true_sharpe_annual}")
    for _ in range(30):
        mid = 0.5 * (low + high)
        if _oracle_sharpe_annual(mid, kind, periods_per_year, vol) < true_sharpe_annual:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def planted_signal_prices(
    n_obs: int,
    *,
    true_sharpe_annual: float,
    kind: Literal["trend", "mean_reversion"] = "trend",
    periods_per_year: int = 252,
    rng: RNGLike = None,
) -> tuple[F64, str]:
    """Return prices with a signal that a named oracle strategy can harvest.

    ``trend`` adds an AR(1) component with persistence 0.98 to the returns. ``mean_reversion``
    adds a stationary AR(1) deviation with persistence 0.90 to the log price. The amplitude is
    calibrated on a long seeded path, so that the oracle's Sharpe ratio before costs equals
    ``true_sharpe_annual``. Calibration runs once per target and is then cached.

    Parameters
    ----------
    n_obs : int
        Number of prices.
    true_sharpe_annual : float
        True annual Sharpe ratio of the oracle, at least 0.
    kind : {"trend", "mean_reversion"}, default "trend"
        Type of signal.
    periods_per_year : int, default 252
        Periods per year.
    rng : numpy.random.Generator or int or None, optional
        Source of randomness of the returned path.

    Returns
    -------
    prices : numpy.ndarray
        ``n_obs`` prices.
    oracle : str
        Identifier of the oracle strategy, from ``StrategySpec.spec_id``.

    Raises
    ------
    ValueError
        If an argument is out of range or the target cannot be reached.

    Examples
    --------
    >>> prices, oracle = planted_signal_prices(50, true_sharpe_annual=0.0, rng=0)
    >>> prices.size, oracle.split(":")[0]
    (50, 'ma_cross')
    """
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    if kind not in ("trend", "mean_reversion"):
        raise ValueError(f"unknown kind {kind!r}")
    if true_sharpe_annual < 0.0:
        raise ValueError("true_sharpe_annual must not be negative")
    strength = 0.0
    if true_sharpe_annual > 0.0:
        strength = _calibrate_strength(true_sharpe_annual, kind, periods_per_year)
    vol = 0.16 / math.sqrt(periods_per_year)
    prices = _signal_prices(n_obs, strength, kind, vol, np.random.default_rng(rng))
    oracle = _TREND_ORACLE if kind == "trend" else _REVERSION_ORACLE
    return prices, oracle.spec_id()


def block_bootstrap_returns(
    returns: ArrayLike,
    n_obs: int,
    *,
    mean_block: float | None = None,
    demean: bool = True,
    rng: RNGLike = None,
) -> F64:
    """Return a stationary bootstrap of a real return series.

    Parameters
    ----------
    returns : array_like
        One-dimensional returns.
    n_obs : int
        Length of the resampled series.
    mean_block : float, optional
        Mean block length. Defaults to :func:`~overfitlab.bootstrap.optimal_block_length`.
    demean : bool, default True
        Remove the sample mean first, which imposes the null of no edge.
    rng : numpy.random.Generator or int or None, optional
        Source of randomness.

    Returns
    -------
    numpy.ndarray
        ``n_obs`` resampled returns.

    Raises
    ------
    ValueError
        If ``n_obs`` is below 1 or the input has NaN or inf.

    Examples
    --------
    >>> block_bootstrap_returns([0.01, -0.02, 0.03, 0.0], 6, mean_block=2.0, rng=0).shape
    (6,)
    """
    _check_size(n_obs)
    series = as_vector(returns, name="returns")
    if demean:
        series = series - series.mean()
    block = optimal_block_length(series) if mean_block is None else mean_block
    generator = np.random.default_rng(rng)
    # The stationary bootstrap resamples in place; tile the indices for a longer series.
    n_rows = math.ceil(n_obs / series.size)
    indices = stationary_bootstrap_indices(series.size, n_rows, mean_block=block, rng=generator)
    return series[indices.ravel()[:n_obs]]
