import json
import math

import numpy as np
import pytest

import overfitlab as ol
from overfitlab import synth


def test_gaussian_noise_shape_and_volatility():
    r = ol.gaussian_noise(50_000, 3, vol_annual=0.2, periods_per_year=252, rng=0)
    assert r.shape == (50_000, 3)
    assert r.std() * math.sqrt(252) == pytest.approx(0.2, rel=0.01)
    assert abs(r.mean()) < 3 * 0.2 / math.sqrt(252) / math.sqrt(r.size)


def test_student_t_noise_is_scaled_and_fat_tailed():
    r = ol.student_t_noise(400_000, df=6.0, vol_annual=0.2, rng=0)
    assert r.std() * math.sqrt(252) == pytest.approx(0.2, rel=0.02)
    assert ol.return_moments(r[:, 0]).kurtosis > 4.0


def test_student_t_rejects_low_df():
    with pytest.raises(ValueError, match="df"):
        ol.student_t_noise(10, df=2.0)


def test_gbm_starts_at_s0_with_the_right_length():
    prices = ol.gbm_prices(50, s0=25.0, rng=0)
    assert prices.size == 50
    assert prices[0] == pytest.approx(25.0)


def test_gbm_drift_and_volatility():
    prices = ol.gbm_prices(500_000, mu_annual=0.1, vol_annual=0.2, rng=0)
    log_returns = np.diff(np.log(prices))
    assert log_returns.std() * math.sqrt(252) == pytest.approx(0.2, rel=0.01)
    assert log_returns.mean() * 252 == pytest.approx(0.1 - 0.02, abs=0.02)


def test_gbm_rejects_bad_price():
    with pytest.raises(ValueError, match="s0"):
        ol.gbm_prices(10, s0=0.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: ol.gaussian_noise(0),
        lambda: ol.gaussian_noise(5, 0),
        lambda: ol.student_t_noise(0),
        lambda: ol.gbm_prices(0),
        lambda: ol.planted_signal_matrix(0, 1, true_sharpe_annual=1.0),
        lambda: ol.block_bootstrap_returns([0.1, 0.2], 0),
    ],
)
def test_sizes_are_validated(call):
    with pytest.raises(ValueError, match="at least 1"):
        call()


def test_seeded_generators_repeat():
    assert np.array_equal(ol.gaussian_noise(10, rng=1), ol.gaussian_noise(10, rng=1))
    assert np.array_equal(ol.gbm_prices(10, rng=1), ol.gbm_prices(10, rng=1))


def test_planted_matrix_truth_flags_and_sharpes():
    result = ol.planted_signal_matrix(100, 6, true_sharpe_annual=2.0, n_true=2, rng=0)
    assert result.returns.shape == (100, 6)
    assert result.is_true.tolist() == [True, True, False, False, False, False]
    assert result.true_sharpe_annual.tolist() == [2.0, 2.0, 0.0, 0.0, 0.0, 0.0]


def test_planted_matrix_recovers_the_true_sharpe():
    result = ol.planted_signal_matrix(1_000_000, 2, true_sharpe_annual=1.5, rng=0)
    realised = ol.sharpe_ratio(result.returns) * math.sqrt(252)
    assert realised[0] == pytest.approx(1.5, abs=0.1)
    assert realised[1] == pytest.approx(0.0, abs=0.1)


def test_planted_matrix_has_the_requested_correlation():
    result = ol.planted_signal_matrix(200_000, 3, true_sharpe_annual=0.0, corr=0.4, rng=0)
    correlation = np.corrcoef(result.returns, rowvar=False)
    assert correlation[0, 1] == pytest.approx(0.4, abs=0.02)


@pytest.mark.parametrize(
    ("kwargs", "message"), [({"n_true": 9}, "n_true"), ({"corr": 1.0}, "corr")]
)
def test_planted_matrix_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ol.planted_signal_matrix(10, 3, true_sharpe_annual=1.0, **kwargs)


def oracle_sharpe(prices, oracle):
    family, params = oracle.split(":", 1)
    spec = ol.StrategySpec(family, json.loads(params))
    returns = ol.run_strategies(prices, [spec], cost_bps=0.0)[:, 0]
    return ol.sharpe_ratio(returns) * math.sqrt(252)


@pytest.mark.parametrize("kind", ["trend", "mean_reversion"])
def test_planted_prices_give_the_oracle_its_sharpe(kind):
    prices, oracle = ol.planted_signal_prices(200_000, true_sharpe_annual=1.5, kind=kind, rng=11)
    assert oracle_sharpe(prices, oracle) == pytest.approx(1.5, abs=0.15)


@pytest.mark.parametrize("kind", ["trend", "mean_reversion"])
def test_zero_sharpe_planted_prices_are_noise(kind):
    prices, oracle = ol.planted_signal_prices(200_000, true_sharpe_annual=0.0, kind=kind, rng=11)
    assert oracle_sharpe(prices, oracle) == pytest.approx(0.0, abs=0.15)


def test_planted_prices_are_positive_and_seeded():
    first, _ = ol.planted_signal_prices(100, true_sharpe_annual=1.0, rng=3)
    second, _ = ol.planted_signal_prices(100, true_sharpe_annual=1.0, rng=3)
    assert (first > 0).all()
    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"n_obs": 1, "true_sharpe_annual": 1.0}, "n_obs"),
        ({"n_obs": 5, "true_sharpe_annual": -1.0}, "negative"),
        ({"n_obs": 5, "true_sharpe_annual": 1.0, "kind": "other"}, "kind"),
    ],
)
def test_planted_prices_validation(args, message):
    with pytest.raises(ValueError, match=message):
        ol.planted_signal_prices(args.pop("n_obs"), **args)


def test_unreachable_target_is_reported(monkeypatch):
    monkeypatch.setattr(synth, "_oracle_sharpe_annual", lambda *_: 0.0)
    synth._calibrate_strength.cache_clear()
    with pytest.raises(ValueError, match="cannot reach"):
        synth._calibrate_strength(99.0, "trend", 252)
    synth._calibrate_strength.cache_clear()


def test_block_bootstrap_demeans_by_default():
    source = np.random.default_rng(0).normal(0.01, 0.01, 2000)
    resampled = ol.block_bootstrap_returns(source, 100_000, mean_block=10.0, rng=0)
    assert abs(resampled.mean()) < 3e-4


def test_block_bootstrap_can_keep_the_mean():
    source = np.random.default_rng(0).normal(0.01, 0.01, 2000)
    resampled = ol.block_bootstrap_returns(source, 100_000, mean_block=10.0, demean=False, rng=0)
    assert resampled.mean() == pytest.approx(source.mean(), abs=5e-4)


def test_block_bootstrap_keeps_dependence():
    source = np.zeros(5000)
    rng = np.random.default_rng(0)
    for t in range(1, source.size):
        source[t] = 0.8 * source[t - 1] + rng.standard_normal()
    resampled = ol.block_bootstrap_returns(source, 5000, rng=1)
    assert np.corrcoef(resampled[:-1], resampled[1:])[0, 1] > 0.4


def test_block_bootstrap_length_may_exceed_the_sample():
    assert ol.block_bootstrap_returns(np.arange(20.0), 55, rng=0).size == 55
