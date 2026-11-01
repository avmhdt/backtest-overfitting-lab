import json
from collections import Counter

import numpy as np
import pytest

import overfitlab as ol
from overfitlab import factory
from overfitlab.factory import StrategySpec, strategy_positions
from overfitlab_study.config import load_study_config


def sign(x):
    return np.sign(np.nan_to_num(x))


def reference_sma(close, window):
    out = np.full(close.size, np.nan)
    for t in range(window - 1, close.size):
        out[t] = close[t - window + 1 : t + 1].mean()
    return out


def reference_ma(close, fast, slow):
    return sign(reference_sma(close, fast) - reference_sma(close, slow))


def reference_channel(close, window, kind):
    out = np.full(close.size, np.nan)
    for t in range(window, close.size):
        out[t] = kind(close[t - window : t])
    return out


def reference_breakout(close, entry, exit_length):
    high, low = reference_channel(close, entry, np.max), reference_channel(close, entry, np.min)
    exit_high = reference_channel(close, exit_length, np.max)
    exit_low = reference_channel(close, exit_length, np.min)
    out = np.zeros(close.size)
    position = 0.0
    for t in range(close.size):
        if position > 0 and close[t] < exit_low[t]:
            position = 0.0
        elif position < 0 and close[t] > exit_high[t]:
            position = 0.0
        if close[t] > high[t]:
            position = 1.0
        elif close[t] < low[t]:
            position = -1.0
        out[t] = position
    return out


def reference_reversion(close, lookback, entry_z, exit_z):
    out = np.zeros(close.size)
    position = 0.0
    for t in range(lookback - 1, close.size):
        window = close[t - lookback + 1 : t + 1]
        std = window.std(ddof=1)
        z = (close[t] - window.mean()) / std if std > 1e-9 else 0.0
        if abs(z) < exit_z:
            position = 0.0
        if z > entry_z:
            position = -1.0
        elif z < -entry_z:
            position = 1.0
        out[t] = position
    return out


def close_path(n_obs=700, seed=3):
    return ol.gbm_prices(n_obs, vol_annual=0.3, rng=seed)


def test_default_grid_has_the_spec_counts():
    counts = Counter(spec.family for spec in ol.strategy_grid())
    assert counts == {"ma_cross": 288, "breakout": 200, "mean_reversion": 546}


def test_default_grid_is_unique():
    assert len({spec.spec_id() for spec in ol.strategy_grid()}) == 1034


def test_grid_from_config():
    grid = ol.strategy_grid({"ma_cross": {"fast": [2, 3], "slow": [10], "mode": ["long_only"]}})
    assert len(grid) == 2
    assert grid[0].params == {"fast": 2, "slow": 10, "mode": "long_only"}


def test_grid_matches_the_study_config():
    assert len(ol.strategy_grid(load_study_config()["factory"]["grid"])) == 1034


def test_grid_rejects_unknown_family():
    with pytest.raises(ValueError, match="family"):
        ol.strategy_grid({"momentum": {"n": [1]}})


def test_params_json_is_canonical():
    spec = StrategySpec("breakout", {"mode": "long_only", "entry": 20, "exit_frac": 0.5})
    assert spec.params_json() == '{"entry":20,"exit_frac":0.5,"mode":"long_only"}'
    assert json.loads(spec.spec_id().split(":", 1)[1])["entry"] == 20


@pytest.mark.parametrize(("fast", "slow"), [(2, 60), (5, 100), (50, 250)])
def test_ma_kernel_matches_the_reference(fast, slow):
    close = close_path()
    spec = StrategySpec("ma_cross", {"fast": fast, "slow": slow, "mode": "long_short"})
    assert np.array_equal(strategy_positions(close, [spec])[:, 0], reference_ma(close, fast, slow))


@pytest.mark.parametrize(("entry", "frac"), [(10, 0.25), (20, 1.0), (50, 0.5), (130, 0.25)])
def test_breakout_kernel_matches_the_reference(entry, frac):
    close = close_path()
    spec = StrategySpec("breakout", {"entry": entry, "exit_frac": frac, "mode": "long_short"})
    expected = reference_breakout(close, entry, max(1, round(frac * entry)))
    assert np.array_equal(strategy_positions(close, [spec])[:, 0], expected)


@pytest.mark.parametrize(
    ("lookback", "entry_z", "exit_z"), [(5, 1.0, 0.0), (20, 1.5, 0.25), (60, 2.0, 0.5)]
)
def test_reversion_kernel_matches_the_reference(lookback, entry_z, exit_z):
    close = close_path()
    params = {"lookback": lookback, "entry_z": entry_z, "exit_z": exit_z, "mode": "long_short"}
    spec = StrategySpec("mean_reversion", params)
    expected = reference_reversion(close, lookback, entry_z, exit_z)
    assert np.array_equal(strategy_positions(close, [spec])[:, 0], expected)


@pytest.mark.parametrize("window", [1, 2, 7, 60])
def test_rolling_sum_kernel_matches_the_reference(window):
    values = close_path(100)
    expected = reference_sma(values, window) * window
    assert np.allclose(factory._rolling_sum(values, window), expected, equal_nan=True)


@pytest.mark.parametrize(("window", "sign_", "kind"), [(5, 1.0, np.max), (9, -1.0, np.min)])
def test_rolling_extreme_kernel_matches_the_reference(window, sign_, kind):
    values = close_path(100)
    expected = reference_channel(values, window, kind)
    assert np.allclose(factory._rolling_extreme(values, window, sign_), expected, equal_nan=True)


def test_net_returns_kernel_matches_the_reference():
    positions = np.array([[0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
    returns = np.array([0.01, 0.02, -0.03])
    out = factory._net_returns(positions, returns, 0.001)
    expected = np.array([[0.0, 0.009], [0.019, 0.02], [0.03 - 0.002, -0.001]])
    assert np.allclose(out, expected)


def test_long_only_clips_short_positions():
    close = close_path()
    short = StrategySpec("ma_cross", {"fast": 5, "slow": 60, "mode": "long_short"})
    long = StrategySpec("ma_cross", {"fast": 5, "slow": 60, "mode": "long_only"})
    positions = strategy_positions(close, [short, long])
    assert np.array_equal(positions[:, 1], np.maximum(positions[:, 0], 0.0))


def test_all_default_positions_are_minus_one_zero_or_one():
    positions = strategy_positions(close_path(), ol.strategy_grid())
    assert set(np.unique(positions)) <= {-1.0, 0.0, 1.0}


def test_position_earns_the_next_bar():
    close = np.array([100.0, 100.0, 110.0, 121.0, 121.0])
    spec = StrategySpec("ma_cross", {"fast": 1, "slow": 2, "mode": "long_short"})
    out = ol.run_strategies(close, [spec], cost_bps=0.0)[:, 0]
    # Flat until the close of bar 2, which earns bar 3; the first move is not captured.
    assert out.tolist() == pytest.approx([0.0, 0.0, 0.1, 0.0])


def test_cost_is_charged_per_unit_of_turnover():
    close = np.array([100.0, 110.0, 100.0, 110.0])
    spec = StrategySpec("ma_cross", {"fast": 1, "slow": 2, "mode": "long_short"})
    gross = ol.run_strategies(close, [spec], cost_bps=0.0)[:, 0]
    net = ol.run_strategies(close, [spec], cost_bps=10.0)[:, 0]
    assert (gross - net).tolist() == pytest.approx([0.0, 0.001, 0.002])


def test_output_is_t_by_n():
    assert ol.run_strategies(close_path(301), ol.strategy_grid()).shape == (300, 1034)


def test_too_little_history_stays_flat():
    close = close_path(30)
    spec = StrategySpec("ma_cross", {"fast": 5, "slow": 60, "mode": "long_short"})
    assert (ol.run_strategies(close, [spec]) == 0.0).all()


def test_flat_prices_stay_flat():
    spec = StrategySpec(
        "mean_reversion", {"lookback": 5, "entry_z": 1.0, "exit_z": 0.0, "mode": "long_short"}
    )
    assert (strategy_positions(np.full(50, 100.0), [spec]) == 0.0).all()


@pytest.mark.parametrize(
    ("close", "kwargs", "message"),
    [
        ([1.0, -1.0], {}, "positive"),
        ([1.0], {}, "two prices"),
        ([1.0, 2.0], {"cost_bps": -1.0}, "cost_bps"),
    ],
)
def test_run_strategies_validation(close, kwargs, message):
    with pytest.raises(ValueError, match=message):
        ol.run_strategies(close, [], **kwargs)


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="family"):
        strategy_positions([1.0, 2.0, 3.0], [StrategySpec("other", {})])
