"""Stationary bootstrap indices and automatic block length.

Indices are generated with a NumPy ``Generator``, so results are the same on every OS, and
are passed into the tests as plain arrays.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from overfitlab._arrays import as_vector

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
RNGLike = np.random.Generator | int | None


def stationary_bootstrap_indices(
    n_obs: int, n_boot: int, *, mean_block: float, rng: RNGLike = None
) -> I64:
    """Return indices of the stationary bootstrap of Politis and Romano (1994).

    Each position starts a new block with probability ``1 / mean_block`` and otherwise
    continues the previous block, wrapping around the end of the sample. Block lengths are
    geometric with mean ``mean_block``.

    Parameters
    ----------
    n_obs : int
        Observations in the sample and in each resample.
    n_boot : int
        Number of resamples.
    mean_block : float
        Mean block length, at least 1.
    rng : numpy.random.Generator or int or None, optional
        Source of randomness.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(n_boot, n_obs)``.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> stationary_bootstrap_indices(5, 2, mean_block=2.0, rng=0).shape
    (2, 5)
    """
    if n_obs < 1 or n_boot < 1:
        raise ValueError("n_obs and n_boot must be at least 1")
    if mean_block < 1.0:
        raise ValueError("mean_block must be at least 1")
    generator = np.random.default_rng(rng)
    starts_block = generator.random((n_boot, n_obs)) < 1.0 / mean_block
    starts_block[:, 0] = True
    starts = generator.integers(0, n_obs, size=(n_boot, n_obs), dtype=np.int64)
    positions = np.arange(n_obs, dtype=np.int64)
    latest = np.maximum.accumulate(np.where(starts_block, positions, 0), axis=1)
    origin = np.take_along_axis(starts, latest, axis=1)
    return (origin + positions - latest) % n_obs


def optimal_block_length(x: ArrayLike) -> float:
    r"""Return the stationary bootstrap block length of Politis and White (2004).

    Uses the flat-top lag window and the correction of Patton, Politis and White (2009):
    :math:`b = (2\hat{G}^2/\hat{D})^{1/3} n^{1/3}`, with
    :math:`\hat{D} = 2\hat{\sigma}^4`. The result is capped at the sample's upper bound
    :math:`\min(3\sqrt{n}, n/3)` and floored at 1.

    Parameters
    ----------
    x : array_like
        One-dimensional series.

    Returns
    -------
    float
        Mean block length.

    Raises
    ------
    ValueError
        If there are fewer than nine observations.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> round(optimal_block_length(rng.standard_normal(1000)), 1) < 3.0
    True
    """
    series = as_vector(x, name="x")
    n_obs = series.size
    if n_obs < 9:
        raise ValueError("at least 9 observations are needed")
    centred = series - series.mean()
    variance = float(np.dot(centred, centred)) / n_obs
    if variance <= 0.0:
        return 1.0

    significance_run = max(5, math.ceil(math.log10(n_obs)))
    max_lag = math.ceil(math.sqrt(n_obs)) + significance_run
    max_lag = min(max_lag, n_obs - 1)
    autocov = np.array(
        [float(np.dot(centred[:-k], centred[k:])) / n_obs for k in range(1, max_lag + 1)]
    )
    rho = np.abs(autocov) / variance

    threshold = 2.0 * math.sqrt(math.log10(n_obs) / n_obs)
    lag = 0
    for k in range(rho.size - significance_run + 1):
        if (rho[k : k + significance_run] < threshold).all():
            lag = k
            break
    else:
        significant = np.flatnonzero(rho > threshold)
        lag = int(significant[-1]) + 1 if significant.size else 0
    window_end = min(2 * max(lag, 1), max_lag)

    lags = np.arange(1, window_end + 1, dtype=np.float64)
    ratio = lags / window_end
    flat_top = np.where(ratio <= 0.5, 1.0, 2.0 * (1.0 - ratio))
    weighted = flat_top * autocov[:window_end]
    g_hat = 2.0 * float(np.sum(weighted * lags))
    long_run = variance + 2.0 * float(np.sum(weighted))
    d_hat = 2.0 * long_run**2
    if g_hat == 0.0 or d_hat == 0.0:
        return 1.0
    length = (2.0 * g_hat**2 / d_hat) ** (1.0 / 3.0) * n_obs ** (1.0 / 3.0)
    return float(min(max(length, 1.0), 3.0 * math.sqrt(n_obs), n_obs / 3.0))
