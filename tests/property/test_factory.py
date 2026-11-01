import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import overfitlab as ol
from overfitlab.factory import strategy_positions

SPECS = ol.strategy_grid(
    {
        "ma_cross": {"fast": [2, 10], "slow": [20, 60], "mode": ["long_short", "long_only"]},
        "breakout": {"entry": [10, 30], "exit_frac": [0.5, 1.0], "mode": ["long_short"]},
        "mean_reversion": {
            "lookback": [5, 20],
            "entry_z": [1.0, 2.0],
            "exit_z": [0.0, 0.5],
            "mode": ["long_short"],
        },
    }
)


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 10_000), st.integers(70, 249), st.floats(0.5, 2.0))
def test_positions_ignore_future_prices(seed, cutoff, shock):
    close = ol.gbm_prices(250, vol_annual=0.3, rng=seed)
    changed = close.copy()
    changed[cutoff + 1 :] *= shock
    original = strategy_positions(close, SPECS)
    perturbed = strategy_positions(changed, SPECS)
    assert np.array_equal(original[: cutoff + 1], perturbed[: cutoff + 1])
