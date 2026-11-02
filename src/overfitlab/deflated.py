"""Deflated Sharpe Ratio and minimum backtest length."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from overfitlab._arrays import as_matrix
from overfitlab._normal import EULER_GAMMA, norm_cdf, upper_quantile
from overfitlab.results import DSRResult
from overfitlab.sharpe import return_moments, sharpe_ratio, sharpe_std


def _max_order_statistic(n_trials: float) -> float:
    """Return the expected maximum of ``n_trials`` standard normals, per the paper's formula."""
    if n_trials < 1.0:
        raise ValueError("n_trials must be at least 1")
    if n_trials == 1.0:
        return 0.0
    return (1.0 - EULER_GAMMA) * upper_quantile(1.0 / n_trials) + EULER_GAMMA * upper_quantile(
        1.0 / (n_trials * math.e)
    )


def expected_max_sharpe(n_trials: float, var_trials: float, *, mean_trials: float = 0.0) -> float:
    r"""Return the expected maximum Sharpe ratio over independent trials.

    :math:`E[\max] \approx \mu + \sqrt{V}\,((1-\gamma)\Phi^{-1}(1-1/N)
    + \gamma\Phi^{-1}(1-1/(Ne)))`, from Bailey and Lopez de Prado (2014), with
    :math:`\gamma` the Euler-Mascheroni constant. The result has the units of the trial
    Sharpe ratios: per-period inputs give a per-period result.

    Parameters
    ----------
    n_trials : float
        Number of independent trials ``N``, possibly non-integer. ``N = 1`` gives
        ``mean_trials``.
    var_trials : float
        Variance ``V`` of the trial Sharpe ratios.
    mean_trials : float, default 0.0
        Mean of the trial Sharpe ratios.

    Returns
    -------
    float
        Expected maximum Sharpe ratio.

    Raises
    ------
    ValueError
        If ``n_trials < 1`` or ``var_trials < 0``.

    Examples
    --------
    >>> round(expected_max_sharpe(100, 0.5 / 250), 4)
    0.1132
    """
    if var_trials < 0.0:
        raise ValueError("var_trials must not be negative")
    return mean_trials + math.sqrt(var_trials) * _max_order_statistic(n_trials)


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    *,
    n_trials: float,
    var_trials: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> DSRResult:
    r"""Return the Deflated Sharpe Ratio.

    :math:`DSR = PSR(SR_0)` with :math:`SR_0` from :func:`expected_max_sharpe`, from
    Bailey and Lopez de Prado (2014). ``sr`` and ``var_trials`` are per period; kurtosis
    is non-excess.

    Parameters
    ----------
    sr : float
        Per-period Sharpe ratio of the selected trial.
    n_obs : int
        Number of observations ``T``.
    n_trials : float
        Number of independent trials ``N``, possibly non-integer.
    var_trials : float
        Variance ``V`` of the per-period trial Sharpe ratios.
    skew : float, default 0.0
        Skewness of the selected trial.
    kurtosis : float, default 3.0
        Non-excess kurtosis of the selected trial.

    Returns
    -------
    DSRResult
        DSR, threshold ``sr0`` and the score ``z``.

    Raises
    ------
    ValueError
        If an argument is out of range or the variance term is not positive.

    Examples
    --------
    >>> r = deflated_sharpe_ratio(2.5 / 250**0.5, 1250, n_trials=100, var_trials=0.5 / 250,
    ...                           skew=-3.0, kurtosis=10.0)
    >>> round(r.dsr, 4)
    0.9004
    """
    sr0 = expected_max_sharpe(n_trials, var_trials)
    z = (sr - sr0) / sharpe_std(sr, n_obs, skew=skew, kurtosis=kurtosis)
    return DSRResult(
        dsr=norm_cdf(z),
        sr=sr,
        sr0=sr0,
        z=z,
        n_trials=n_trials,
        var_trials=var_trials,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurtosis,
    )


def dsr_from_returns(
    returns: ArrayLike, *, selected: int | None = None, n_trials: float | None = None
) -> DSRResult:
    """Return the DSR of a ``T x N`` return matrix, from Bailey and Lopez de Prado (2014).

    ``N`` defaults to the number of columns and ``V`` is the sample variance of the
    per-period column Sharpe ratios. With a single column, ``V`` is the null variance of
    the estimator, ``1 / (T - 1)``, as in the trial-log contract.

    Parameters
    ----------
    returns : array_like
        ``T x N`` matrix of per-period returns.
    selected : int, optional
        Column of the selected trial. Defaults to the highest Sharpe ratio.
    n_trials : float, optional
        Number of trials, for example an effective count. Defaults to ``N``.

    Returns
    -------
    DSRResult
        DSR of the selected trial.

    Raises
    ------
    ValueError
        If the input has NaN or inf or ``selected`` is out of range.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> r = rng.normal(0.0, 0.01, (500, 20))
    >>> 0.0 <= dsr_from_returns(r).dsr <= 1.0
    True
    """
    matrix = as_matrix(returns)
    srs = np.atleast_1d(sharpe_ratio(matrix))
    n_cols = srs.size
    index = int(np.argmax(srs)) if selected is None else selected
    if not 0 <= index < n_cols:
        raise ValueError(f"selected must be in [0, {n_cols})")
    variance = float(np.var(srs, ddof=1)) if n_cols > 1 else 1.0 / (matrix.shape[0] - 1)
    moments = return_moments(matrix[:, index])
    return deflated_sharpe_ratio(
        moments.sr,
        moments.n_obs,
        n_trials=float(n_cols) if n_trials is None else n_trials,
        var_trials=variance,
        skew=moments.skew,
        kurtosis=moments.kurtosis,
    )


def min_backtest_length(n_trials: float, *, max_sharpe_annual: float = 1.0) -> float:
    r"""Return the Minimum Backtest Length, in years.

    :math:`MinBTL \approx (((1-\gamma)\Phi^{-1}(1-1/N)
    + \gamma\Phi^{-1}(1-1/(Ne))) / E[\max])^2`, from Bailey et al. (2014, eq. 3.2): the
    years needed so that the expected maximum annual Sharpe ratio of ``N`` skill-less
    independent trials does not exceed ``max_sharpe_annual``.

    Parameters
    ----------
    n_trials : float
        Number of independent trials ``N``.
    max_sharpe_annual : float, default 1.0
        Target expected maximum annual Sharpe ratio.

    Returns
    -------
    float
        Years of backtest.

    Raises
    ------
    ValueError
        If ``n_trials < 1`` or ``max_sharpe_annual`` is not positive.

    Examples
    --------
    >>> round(min_backtest_length(45), 1)
    5.0
    """
    if max_sharpe_annual <= 0.0:
        raise ValueError("max_sharpe_annual must be positive")
    return (_max_order_statistic(n_trials) / max_sharpe_annual) ** 2
