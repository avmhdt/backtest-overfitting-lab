"""Normal distribution helpers built on ``scipy.special``."""

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri

EULER_GAMMA: float = 0.5772156649015329


def norm_cdf(x: float) -> float:
    """Return the standard normal CDF at ``x``.

    Parameters
    ----------
    x : float
        Point of evaluation.

    Returns
    -------
    float
        ``Phi(x)``.

    Examples
    --------
    >>> round(norm_cdf(1.96), 3)
    0.975
    """
    return float(ndtr(x))


def norm_ppf(p: float) -> float:
    """Return the standard normal quantile of ``p``.

    Parameters
    ----------
    p : float
        Probability in ``[0, 1]``.

    Returns
    -------
    float
        ``Phi^-1(p)``.

    Examples
    --------
    >>> round(norm_ppf(0.975), 2)
    1.96
    """
    return float(ndtri(p))


def norm_sf(x: float) -> float:
    """Return the upper tail ``1 - Phi(x)`` without losing precision far in the tail.

    Parameters
    ----------
    x : float
        Point of evaluation.

    Returns
    -------
    float
        ``1 - Phi(x)``, computed from the log lower tail of ``-x``.

    Examples
    --------
    >>> norm_sf(10.0) < 1e-22
    True
    """
    return float(np.exp(log_ndtr(-x)))


def upper_quantile(tail: float) -> float:
    """Return ``Phi^-1(1 - tail)``, accurate when ``tail`` is tiny.

    Parameters
    ----------
    tail : float
        Upper-tail probability in ``(0, 1)``.

    Returns
    -------
    float
        Quantile with ``tail`` probability above it.

    Examples
    --------
    >>> round(upper_quantile(0.05), 3)
    1.645
    """
    return -float(ndtri(tail))
