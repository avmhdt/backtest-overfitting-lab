"""Strategy factory: parameter variants of three rule families, on Numba kernels.

A position computed from data up to close ``t`` earns the return from close ``t`` to close
``t + 1``. All positions are -1, 0 or +1. Long-only variants clip the long-short position
at zero.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numba
import numpy as np
from numpy.typing import ArrayLike, NDArray

from overfitlab._arrays import as_vector
from overfitlab.triallog import canonical_params_json

F64 = NDArray[np.float64]
Family = Literal["ma_cross", "breakout", "mean_reversion"]

DEFAULT_GRID: dict[str, dict[str, Sequence[object]]] = {
    "ma_cross": {
        "fast": [2, 3, 5, 8, 10, 13, 15, 20, 25, 30, 40, 50],
        "slow": [60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250],
        "mode": ["long_short", "long_only"],
    },
    "breakout": {
        "entry": list(range(10, 251, 10)),
        "exit_frac": [0.25, 0.5, 0.75, 1.0],
        "mode": ["long_short", "long_only"],
    },
    "mean_reversion": {
        "lookback": [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150],
        "entry_z": [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
        "exit_z": [0.0, 0.25, 0.5],
        "mode": ["long_short", "long_only"],
    },
}


@dataclass(frozen=True)
class StrategySpec:
    """One parameter variant of a rule family.

    Attributes
    ----------
    family : {"ma_cross", "breakout", "mean_reversion"}
        Rule family.
    params : mapping
        Parameters of the family, including ``mode``.
    """

    family: Family
    params: Mapping[str, float | int | str]

    def params_json(self) -> str:
        """Return the canonical JSON of the parameters, as in the trial log.

        Returns
        -------
        str
            Canonical JSON.

        Examples
        --------
        >>> StrategySpec("ma_cross", {"slow": 60, "fast": 5}).params_json()
        '{"fast":5,"slow":60}'
        """
        return canonical_params_json(self.params)

    def spec_id(self) -> str:
        """Return a readable identifier, ``family:params_json``.

        Returns
        -------
        str
            Identifier.
        """
        return f"{self.family}:{self.params_json()}"


def strategy_grid(
    config: Mapping[str, Mapping[str, Sequence[object]]] | None = None,
) -> list[StrategySpec]:
    """Return the variants of a grid, in a stable order.

    Parameters
    ----------
    config : mapping, optional
        Family to parameter to values. Defaults to the 1,034-variant grid: 288 moving-average
        crossovers, 200 breakouts and 546 mean-reversion rules.

    Returns
    -------
    list of StrategySpec
        One spec per combination of parameter values.

    Raises
    ------
    ValueError
        If a family is unknown.

    Examples
    --------
    >>> len(strategy_grid())
    1034
    """
    specs: list[StrategySpec] = []
    for family, parameters in (config or DEFAULT_GRID).items():
        if family not in DEFAULT_GRID:
            raise ValueError(f"unknown family {family!r}")
        names = list(parameters)
        for values in itertools.product(*(parameters[name] for name in names)):
            params = dict(zip(names, values, strict=True))
            specs.append(StrategySpec(family, params))  # type: ignore[arg-type]
    return specs


@numba.njit(cache=True)  # type: ignore[no-untyped-call,untyped-decorator,unused-ignore]
def _rolling_sum(values: F64, window: int) -> F64:
    """Sum of the last ``window`` values at each row; NaN until the window is full."""
    out = np.full(values.size, np.nan)
    total = 0.0
    for t in range(values.size):
        total += values[t]
        if t >= window:
            total -= values[t - window]
        if t >= window - 1:
            out[t] = total
    return out


@numba.njit(cache=True)  # type: ignore[no-untyped-call,untyped-decorator,unused-ignore]
def _rolling_extreme(close: F64, window: int, sign: float) -> F64:
    """Largest of ``sign * close`` over the ``window`` bars before each row."""
    n_obs = close.size
    out = np.full(n_obs, np.nan)
    queue = np.zeros(n_obs, dtype=np.int64)
    head = 0
    tail = 0
    for t in range(1, n_obs):
        value = sign * close[t - 1]
        while tail > head and sign * close[queue[tail - 1]] <= value:
            tail -= 1
        queue[tail] = t - 1
        tail += 1
        if queue[head] < t - window:
            head += 1
        if t >= window:
            out[t] = close[queue[head]]
    return out


@numba.njit(cache=True)  # type: ignore[no-untyped-call,untyped-decorator,unused-ignore]
def _ma_positions(fast_sma: F64, slow_sma: F64) -> F64:
    out = np.zeros(fast_sma.size)
    for t in range(fast_sma.size):
        gap = fast_sma[t] - slow_sma[t]
        if gap > 0.0:
            out[t] = 1.0
        elif gap < 0.0:
            out[t] = -1.0
    return out


@numba.njit(cache=True)  # type: ignore[no-untyped-call,untyped-decorator,unused-ignore]
def _breakout_positions(
    close: F64, entry_high: F64, entry_low: F64, exit_high: F64, exit_low: F64
) -> F64:
    out = np.zeros(close.size)
    position = 0.0
    for t in range(close.size):
        if position > 0.0 and close[t] < exit_low[t]:
            position = 0.0
        elif position < 0.0 and close[t] > exit_high[t]:
            position = 0.0
        if close[t] > entry_high[t]:
            position = 1.0
        elif close[t] < entry_low[t]:
            position = -1.0
        out[t] = position
    return out


@numba.njit(cache=True)  # type: ignore[no-untyped-call,untyped-decorator,unused-ignore]
def _zscore(close: F64, sum_close: F64, sum_squares: F64, window: int) -> F64:
    out = np.full(close.size, np.nan)
    for t in range(window - 1, close.size):
        mean = sum_close[t] / window
        variance = (sum_squares[t] - window * mean * mean) / (window - 1)
        out[t] = (close[t] - mean) / np.sqrt(variance) if variance > 1e-18 else 0.0
    return out


@numba.njit(cache=True)  # type: ignore[no-untyped-call,untyped-decorator,unused-ignore]
def _reversion_positions(z: F64, entry_z: float, exit_z: float) -> F64:
    out = np.zeros(z.size)
    position = 0.0
    for t in range(z.size):
        if np.isnan(z[t]):
            continue
        if abs(z[t]) < exit_z:
            position = 0.0
        if z[t] > entry_z:
            position = -1.0
        elif z[t] < -entry_z:
            position = 1.0
        out[t] = position
    return out


@numba.njit(cache=True)  # type: ignore[no-untyped-call,untyped-decorator,unused-ignore]
def _net_returns(positions: F64, returns: F64, cost: float) -> F64:
    """Net returns, ``positions`` indexed by close and ``returns`` by the bar after it."""
    n_bars, n_strategies = returns.size, positions.shape[1]
    out = np.zeros((n_bars, n_strategies))
    for j in range(n_strategies):
        previous = 0.0
        for t in range(n_bars):
            position = positions[t, j]
            out[t, j] = position * returns[t] - cost * abs(position - previous)
            previous = position
    return out


class _Inputs:
    """Rolling series shared by the variants of one price path."""

    def __init__(self, close: F64) -> None:
        self.close = close
        self.squares = close * close
        self._sums: dict[int, F64] = {}
        self._square_sums: dict[int, F64] = {}
        self._extremes: dict[tuple[int, float], F64] = {}

    def sma(self, window: int) -> F64:
        if window not in self._sums:
            self._sums[window] = _rolling_sum(self.close, window)
        return self._sums[window] / window

    def sum_close(self, window: int) -> F64:
        if window not in self._sums:
            self._sums[window] = _rolling_sum(self.close, window)
        return self._sums[window]

    def sum_squares(self, window: int) -> F64:
        if window not in self._square_sums:
            self._square_sums[window] = _rolling_sum(self.squares, window)
        return self._square_sums[window]

    def extreme(self, window: int, sign: float) -> F64:
        key = (window, sign)
        if key not in self._extremes:
            self._extremes[key] = _rolling_extreme(self.close, window, sign)
        return self._extremes[key]


def _positions(inputs: _Inputs, spec: StrategySpec) -> F64:
    params = spec.params
    if spec.family == "ma_cross":
        fast, slow = int(params["fast"]), int(params["slow"])
        positions = _ma_positions(inputs.sma(fast), inputs.sma(slow))
    elif spec.family == "breakout":
        entry = int(params["entry"])
        exit_length = max(1, round(float(params["exit_frac"]) * entry))
        positions = _breakout_positions(
            inputs.close,
            inputs.extreme(entry, 1.0),
            inputs.extreme(entry, -1.0),
            inputs.extreme(exit_length, 1.0),
            inputs.extreme(exit_length, -1.0),
        )
    elif spec.family == "mean_reversion":
        window = int(params["lookback"])
        z = _zscore(inputs.close, inputs.sum_close(window), inputs.sum_squares(window), window)
        positions = _reversion_positions(z, float(params["entry_z"]), float(params["exit_z"]))
    else:
        raise ValueError(f"unknown family {spec.family!r}")
    if params.get("mode") == "long_only":
        positions = np.maximum(positions, 0.0)
    return positions


def strategy_positions(close: ArrayLike, specs: Sequence[StrategySpec]) -> F64:
    """Return the position each variant holds after each close.

    Parameters
    ----------
    close : array_like
        Positive closing prices, one-dimensional.
    specs : sequence of StrategySpec
        Variants to run.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_close, n_specs)`` with values -1, 0 and +1.

    Raises
    ------
    ValueError
        If prices are not positive or a family is unknown.

    Examples
    --------
    >>> spec = StrategySpec("ma_cross", {"fast": 1, "slow": 2, "mode": "long_short"})
    >>> strategy_positions([1.0, 2.0, 1.0], [spec])[:, 0].tolist()
    [0.0, 1.0, -1.0]
    """
    prices = as_vector(close, name="close")
    if (prices <= 0.0).any():
        raise ValueError("close must be positive")
    inputs = _Inputs(prices)
    out = np.zeros((prices.size, len(specs)))
    for j, spec in enumerate(specs):
        out[:, j] = _positions(inputs, spec)
    return out


def run_strategies(
    close: ArrayLike, specs: Sequence[StrategySpec], *, cost_bps: float = 1.0
) -> F64:
    r"""Return the net returns of each variant on one price path.

    The position after close ``t - 1`` earns the simple return from close ``t - 1`` to close
    ``t``, less :math:`c\,|\Delta position|` for the trade made at close ``t - 1``.

    Parameters
    ----------
    close : array_like
        Positive closing prices, ``T + 1`` values.
    specs : sequence of StrategySpec
        Variants to run.
    cost_bps : float, default 1.0
        Cost in basis points per unit of turnover.

    Returns
    -------
    numpy.ndarray
        ``T x N`` per-period net returns.

    Raises
    ------
    ValueError
        If prices are not positive, ``cost_bps`` is negative or there are under two prices.

    Examples
    --------
    >>> spec = StrategySpec("ma_cross", {"fast": 1, "slow": 2, "mode": "long_short"})
    >>> run_strategies([1.0, 2.0, 4.0], [spec], cost_bps=0.0)[:, 0].tolist()
    [0.0, 1.0]
    """
    if cost_bps < 0.0:
        raise ValueError("cost_bps must not be negative")
    prices = as_vector(close, name="close")
    if prices.size < 2:
        raise ValueError("at least two prices are needed")
    positions = strategy_positions(prices, specs)
    returns = prices[1:] / prices[:-1] - 1.0
    return np.asarray(_net_returns(positions[:-1], returns, cost_bps / 1e4), dtype=np.float64)
