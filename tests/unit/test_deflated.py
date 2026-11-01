import math

import numpy as np
import pytest

import overfitlab as ol


def test_expected_max_is_zero_for_one_trial():
    assert ol.expected_max_sharpe(1, 0.5) == 0.0


def test_expected_max_adds_the_mean():
    assert ol.expected_max_sharpe(1, 0.5, mean_trials=0.2) == 0.2
    base = ol.expected_max_sharpe(50, 0.5)
    assert ol.expected_max_sharpe(50, 0.5, mean_trials=0.2) == pytest.approx(base + 0.2)


def test_expected_max_scales_with_root_variance():
    assert ol.expected_max_sharpe(50, 4.0) == pytest.approx(2 * ol.expected_max_sharpe(50, 1.0))


def test_expected_max_accepts_non_integer_trials():
    low, mid, high = (ol.expected_max_sharpe(n, 1.0) for n in (10, 10.5, 11))
    assert low < mid < high


def test_expected_max_close_to_simulated_maximum():
    # The paper's formula is an approximation, about 1% high at N = 100.
    draws = np.random.default_rng(0).standard_normal((20_000, 100)).max(axis=1)
    assert ol.expected_max_sharpe(100, 1.0) == pytest.approx(draws.mean(), abs=0.03)


@pytest.mark.parametrize("n_trials", [1e2, 1e4, 1e6, 1e7])
def test_expected_max_stable_for_large_trials(n_trials):
    value = ol.expected_max_sharpe(n_trials, 1.0)
    assert math.isfinite(value)
    assert value < math.sqrt(2 * math.log(n_trials))


@pytest.mark.parametrize(
    ("n_trials", "var_trials", "message"), [(0.5, 1.0, "n_trials"), (10, -1.0, "var_trials")]
)
def test_expected_max_argument_validation(n_trials, var_trials, message):
    with pytest.raises(ValueError, match=message):
        ol.expected_max_sharpe(n_trials, var_trials)


def test_dsr_with_one_trial_is_psr_at_zero():
    result = ol.deflated_sharpe_ratio(0.1, 500, n_trials=1, var_trials=0.01)
    assert result.dsr == pytest.approx(ol.probabilistic_sharpe_ratio(0.1, 500))
    assert result.sr0 == 0.0


def test_dsr_result_to_dict():
    result = ol.deflated_sharpe_ratio(0.1, 500, n_trials=10, var_trials=0.01)
    assert result.to_dict()["dsr"] == result.dsr


def test_dsr_from_returns_picks_the_best_column():
    r = np.random.default_rng(0).normal(0.0, 0.01, (400, 6))
    best = int(np.argmax(ol.sharpe_ratio(r)))
    assert ol.dsr_from_returns(r).sr == pytest.approx(ol.sharpe_ratio(r[:, best]))


def test_dsr_from_returns_uses_variance_and_count():
    r = np.random.default_rng(0).normal(0.0, 0.01, (400, 6))
    result = ol.dsr_from_returns(r, selected=2, n_trials=3.5)
    assert result.n_trials == 3.5
    assert result.var_trials == pytest.approx(np.var(ol.sharpe_ratio(r), ddof=1))


def test_dsr_from_returns_single_column_uses_null_variance():
    r = np.random.default_rng(0).normal(0.001, 0.01, 101)
    result = ol.dsr_from_returns(r)
    assert result.var_trials == pytest.approx(0.01)
    assert result.sr0 == 0.0


def test_dsr_from_returns_rejects_bad_selection():
    with pytest.raises(ValueError, match="selected"):
        ol.dsr_from_returns(np.ones((3, 2)) + np.arange(6).reshape(3, 2), selected=5)


def test_min_backtest_length_inverts_expected_max():
    years = ol.min_backtest_length(100, max_sharpe_annual=1.5)
    assert ol.expected_max_sharpe(100, 1.0 / years) == pytest.approx(1.5)


def test_min_backtest_length_respects_the_upper_bound():
    # Bailey et al. (2014, eq. 3.2): MinBTL < 2 ln(N) / E[max]^2.
    assert ol.min_backtest_length(45) < 2 * math.log(45)


def test_min_backtest_length_validation():
    with pytest.raises(ValueError, match="max_sharpe_annual"):
        ol.min_backtest_length(10, max_sharpe_annual=0.0)
