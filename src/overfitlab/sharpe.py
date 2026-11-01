"""Sharpe ratio statistics: moments, annualisation, PSR and minimum track record length.

"Per-period" Sharpe ratios are not annualised. Kurtosis is non-excess (normal = 3)
everywhere, as in Bailey and Lopez de Prado.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from overfitlab._arrays import as_float_array, as_vector
from overfitlab._normal import norm_cdf, upper_quantile
from overfitlab.results import ReturnMoments

F64 = NDArray[np.float64]


def sharpe_ratio(returns: ArrayLike, *, axis: int = 0, ddof: int = 1) -> float | F64:
    r"""Return the per-period Sharpe ratio ``mean / std``.

    A series with zero variance has no measurable Sharpe ratio and yields 0.0, so a
    flat strategy inside a trial matrix does not abort the evaluation.

    Parameters
    ----------
    returns : array_like
        Per-period returns, one- or two-dimensional.
    axis : int, default 0
        Axis of observations.
    ddof : int, default 1
        Delta degrees of freedom of the standard deviation.

    Returns
    -------
    float or numpy.ndarray
        A float for one-dimensional input, otherwise one value per series.

    Raises
    ------
    ValueError
        If there are fewer than two observations or the input has NaN or inf.

    Examples
    --------
    >>> round(sharpe_ratio([0.01, 0.02, 0.03]), 6)
    2.0
    """
    arr = as_float_array(returns, name="returns")
    if arr.shape[axis] < 2:
        raise ValueError("n_obs must be at least 2")
    mean = arr.mean(axis=axis)
    std = arr.std(axis=axis, ddof=ddof)
    # Rounding leaves a constant series with a tiny std, which is still zero variance.
    scale = np.maximum(np.abs(arr).max(axis=axis), 1e-300)
    flat = std <= 1e-14 * scale
    safe_std = np.where(flat, 1.0, std)
    out = np.where(flat, 0.0, mean / safe_std)
    return float(out) if out.ndim == 0 else out


def return_moments(returns: ArrayLike) -> ReturnMoments:
    r"""Return the Sharpe ratio, skewness and non-excess kurtosis of a series.

    Skewness is :math:`m_3 / m_2^{3/2}` and kurtosis :math:`m_4 / m_2^2`, with central
    moments :math:`m_k` averaged over ``n_obs``, without a small-sample correction.

    Parameters
    ----------
    returns : array_like
        Per-period returns, one-dimensional.

    Returns
    -------
    ReturnMoments
        Per-period Sharpe ratio, skewness, kurtosis (normal = 3) and ``n_obs``.

    Raises
    ------
    ValueError
        If there are fewer than two observations or the series is constant.

    Examples
    --------
    >>> m = return_moments([0.01, -0.01, 0.01, -0.01])
    >>> m.skew, m.kurtosis, m.n_obs
    (0.0, 1.0, 4)
    """
    arr = as_vector(returns, name="returns")
    if arr.size < 2:
        raise ValueError("n_obs must be at least 2")
    centred = arr - arr.mean()
    m2 = float(np.mean(centred**2))
    if m2 <= 0.0:
        raise ValueError("returns have zero variance")
    skew = float(np.mean(centred**3)) / m2**1.5
    kurtosis = float(np.mean(centred**4)) / m2**2
    sr = float(sharpe_ratio(arr))
    return ReturnMoments(sr=sr, skew=skew, kurtosis=kurtosis, n_obs=int(arr.size))


def annualize_sharpe(sr: float, periods_per_year: float, *, autocorr: float | None = None) -> float:
    r"""Annualise a per-period Sharpe ratio.

    Without ``autocorr`` this multiplies by :math:`\sqrt{q}`. With ``autocorr`` it applies
    Lo (2002, eq. 22) for AR(1) returns,
    :math:`\eta(q) = \sqrt{q}\,[1 + \frac{2\rho}{1-\rho}(1 - \frac{1-\rho^q}{q(1-\rho)})]^{-1/2}`.

    Parameters
    ----------
    sr : float
        Per-period Sharpe ratio.
    periods_per_year : float
        Periods per year ``q``.
    autocorr : float, optional
        First-order autocorrelation ``rho`` of per-period returns, in ``(-1, 1)``.

    Returns
    -------
    float
        Annual Sharpe ratio.

    Raises
    ------
    ValueError
        If ``periods_per_year`` is not positive or ``autocorr`` is outside ``(-1, 1)``.

    Examples
    --------
    >>> round(annualize_sharpe(1.0, 12), 2)
    3.46
    >>> round(annualize_sharpe(1.0, 12, autocorr=0.2), 2)
    2.88
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if autocorr is None or autocorr == 0.0:
        return sr * math.sqrt(periods_per_year)
    return sr * math.sqrt(periods_per_year) * ar1_scale(autocorr, periods_per_year)


def ar1_scale(autocorr: float, periods_per_year: float) -> float:
    r"""Return Lo's AR(1) correction relative to the IID factor :math:`\sqrt{q}`.

    Parameters
    ----------
    autocorr : float
        First-order autocorrelation ``rho`` in ``(-1, 1)``.
    periods_per_year : float
        Periods per year ``q``.

    Returns
    -------
    float
        :math:`[1 + \frac{2\rho}{1-\rho}(1 - \frac{1-\rho^q}{q(1-\rho)})]^{-1/2}`.

    Raises
    ------
    ValueError
        If ``autocorr`` is outside ``(-1, 1)``.

    Examples
    --------
    >>> round(ar1_scale(0.1, 12), 4)
    0.9122
    """
    if not -1.0 < autocorr < 1.0:
        raise ValueError("autocorr must be in (-1, 1)")
    rho, q = autocorr, periods_per_year
    term = 1.0 + (2.0 * rho / (1.0 - rho)) * (1.0 - (1.0 - rho**q) / (q * (1.0 - rho)))
    return float(term**-0.5)


def _variance_term(sr: float, skew: float, kurtosis: float) -> float:
    if kurtosis < 1.0:
        raise ValueError("kurtosis must be at least 1 (non-excess, normal = 3)")
    term = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2
    if term <= 0.0:
        raise ValueError(
            f"variance term 1 - skew*sr + (kurtosis-1)/4*sr^2 = {term} is not positive"
        )
    return term


def sharpe_std(sr: float, n_obs: int, *, skew: float = 0.0, kurtosis: float = 3.0) -> float:
    r"""Return the standard error of a per-period Sharpe ratio estimate.

    :math:`\sigma(\widehat{SR}) = \sqrt{(1 - \gamma_3 SR + \frac{\gamma_4-1}{4} SR^2)/(T-1)}`,
    from Bailey and Lopez de Prado (2012), with non-excess kurtosis.

    Parameters
    ----------
    sr : float
        Per-period Sharpe ratio.
    n_obs : int
        Number of observations ``T``.
    skew : float, default 0.0
        Skewness.
    kurtosis : float, default 3.0
        Non-excess kurtosis.

    Returns
    -------
    float
        Standard error, per period.

    Raises
    ------
    ValueError
        If ``n_obs < 2``, ``kurtosis < 1`` or the variance term is not positive.

    Examples
    --------
    >>> round(sharpe_std(0.0, 101), 4)
    0.1
    """
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    return math.sqrt(_variance_term(sr, skew, kurtosis) / (n_obs - 1))


def probabilistic_sharpe_ratio(
    sr: float,
    n_obs: int,
    *,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    r"""Return the Probabilistic Sharpe Ratio.

    :math:`PSR(SR^*) = \Phi\big((SR - SR^*)\sqrt{T-1} /
    \sqrt{1 - \gamma_3 SR + \frac{\gamma_4-1}{4} SR^2}\big)`, from Bailey and Lopez de
    Prado (2012). All Sharpe ratios are per period and kurtosis is non-excess.

    Parameters
    ----------
    sr : float
        Per-period Sharpe ratio.
    n_obs : int
        Number of observations ``T``.
    sr_benchmark : float, default 0.0
        Per-period benchmark Sharpe ratio ``SR*``.
    skew : float, default 0.0
        Skewness.
    kurtosis : float, default 3.0
        Non-excess kurtosis.

    Returns
    -------
    float
        Probability that the true Sharpe ratio exceeds ``sr_benchmark``.

    Raises
    ------
    ValueError
        If ``n_obs < 2``, ``kurtosis < 1`` or the variance term is not positive.

    Examples
    --------
    >>> round(probabilistic_sharpe_ratio(0.458, 24), 3)
    0.982
    """
    z = (sr - sr_benchmark) / sharpe_std(sr, n_obs, skew=skew, kurtosis=kurtosis)
    return norm_cdf(z)


def psr_from_returns(returns: ArrayLike, *, sr_benchmark: float = 0.0) -> float:
    """Return the PSR of a return series, using its own moments.

    Parameters
    ----------
    returns : array_like
        Per-period returns, one-dimensional.
    sr_benchmark : float, default 0.0
        Per-period benchmark Sharpe ratio.

    Returns
    -------
    float
        Probabilistic Sharpe Ratio.

    Examples
    --------
    >>> r = [0.02, 0.01, 0.03, -0.01, 0.02, 0.01]
    >>> 0.0 < psr_from_returns(r) < 1.0
    True
    """
    m = return_moments(returns)
    return probabilistic_sharpe_ratio(
        m.sr, m.n_obs, sr_benchmark=sr_benchmark, skew=m.skew, kurtosis=m.kurtosis
    )


def min_track_record_length(
    sr: float,
    *,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    alpha: float = 0.05,
) -> float:
    r"""Return the Minimum Track Record Length, in observations.

    :math:`MinTRL = 1 + (1 - \gamma_3 SR + \frac{\gamma_4-1}{4} SR^2)
    (z_{1-\alpha} / (SR - SR^*))^2`, from Bailey and Lopez de Prado (2012). All Sharpe
    ratios are per period and kurtosis is non-excess.

    Parameters
    ----------
    sr : float
        Per-period Sharpe ratio.
    sr_benchmark : float, default 0.0
        Per-period benchmark Sharpe ratio ``SR*``.
    skew : float, default 0.0
        Skewness.
    kurtosis : float, default 3.0
        Non-excess kurtosis.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    float
        Number of observations, not years.

    Raises
    ------
    ValueError
        If ``sr <= sr_benchmark``, ``alpha`` is outside ``(0, 1)``, ``kurtosis < 1`` or
        the variance term is not positive.

    Examples
    --------
    >>> round(min_track_record_length(2 / 12**0.5, sr_benchmark=1 / 12**0.5) / 12, 2)
    3.24
    """
    if sr <= sr_benchmark:
        raise ValueError("sr must exceed sr_benchmark")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    term = _variance_term(sr, skew, kurtosis)
    return 1.0 + term * (upper_quantile(alpha) / (sr - sr_benchmark)) ** 2
