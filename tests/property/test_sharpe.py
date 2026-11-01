from hypothesis import given
from hypothesis import strategies as st

import overfitlab as ol

SR = st.floats(-0.5, 0.5)
OBS = st.integers(10, 5000)
TRIALS = st.floats(2.0, 1e6)
VARIANCE = st.floats(1e-5, 1.0)


@given(SR, SR, OBS)
def test_psr_increases_in_sr(low, high, n_obs):
    low, high = sorted((low, high))
    assert ol.probabilistic_sharpe_ratio(low, n_obs) <= (
        ol.probabilistic_sharpe_ratio(high, n_obs) + 1e-12
    )


@given(SR, SR, SR, OBS)
def test_psr_decreases_in_benchmark(sr, low, high, n_obs):
    low, high = sorted((low, high))
    assert ol.probabilistic_sharpe_ratio(sr, n_obs, sr_benchmark=low) >= (
        ol.probabilistic_sharpe_ratio(sr, n_obs, sr_benchmark=high) - 1e-12
    )


@given(st.floats(0.01, 0.5), OBS, TRIALS, VARIANCE)
def test_dsr_never_exceeds_psr_at_zero(sr, n_obs, n_trials, variance):
    dsr = ol.deflated_sharpe_ratio(sr, n_obs, n_trials=n_trials, var_trials=variance).dsr
    assert dsr <= ol.probabilistic_sharpe_ratio(sr, n_obs) + 1e-12


@given(TRIALS, TRIALS, VARIANCE)
def test_expected_max_increases_in_trials(low, high, variance):
    low, high = sorted((low, high))
    assert ol.expected_max_sharpe(low, variance) <= ol.expected_max_sharpe(high, variance) + 1e-12


@given(TRIALS, VARIANCE, VARIANCE)
def test_expected_max_increases_in_variance(n_trials, low, high):
    low, high = sorted((low, high))
    assert ol.expected_max_sharpe(n_trials, low) <= ol.expected_max_sharpe(n_trials, high) + 1e-12


@given(st.floats(0.01, 0.5), st.floats(0.01, 0.5))
def test_mintrl_decreases_in_sr(low, high):
    low, high = sorted((low, high))
    assert ol.min_track_record_length(low) >= ol.min_track_record_length(high) - 1e-9
