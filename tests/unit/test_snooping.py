import math

import numpy as np
import pytest

import overfitlab as ol
from overfitlab import snooping


def noise(n_obs=400, n_series=8, seed=0):
    return np.random.default_rng(seed).normal(0.0, 0.01, (n_obs, n_series))


def planted(n_obs=800, n_series=8, seed=0):
    d = noise(n_obs, n_series, seed)
    d[:, 0] += 0.006
    return d


def test_bootstrap_means_kernel_matches_the_reference():
    values = noise(50, 3)
    indices = ol.stationary_bootstrap_indices(50, 4, mean_block=3.0, rng=0)
    expected = np.array([values[row].mean(axis=0) for row in indices])
    assert snooping._bootstrap_means(values, indices) == pytest.approx(expected, abs=1e-14)


def test_long_run_variance_matches_the_direct_sum():
    values = noise(120, 3)
    n_obs = values.shape[0]
    block = 5.0
    q = 1.0 / block
    demeaned = values - values.mean(axis=0)
    expected = (demeaned**2).sum(axis=0) / n_obs
    for i in range(1, n_obs):
        kappa = (1 - i / n_obs) * (1 - q) ** i + (i / n_obs) * (1 - q) ** (n_obs - i)
        expected += 2 * kappa * (demeaned[:-i] * demeaned[i:]).sum(axis=0) / n_obs
    assert snooping._long_run_variance(values, block) == pytest.approx(expected, rel=1e-9)


def test_flat_column_has_infinite_scale():
    values = noise(100, 2)
    values[:, 1] = 0.0
    assert math.isinf(snooping._long_run_variance(values, 3.0)[1])


def test_reality_check_statistic():
    d = noise()
    result = ol.reality_check(d, n_boot=100, block_length=2.0, rng=0)
    assert result.stat == pytest.approx(math.sqrt(400) * d.mean(axis=0).max())
    assert result.n_boot == 100
    assert result.block_length == 2.0


def test_reality_check_is_reproducible():
    d = noise()
    first = ol.reality_check(d, n_boot=200, block_length=2.0, rng=1)
    assert first == ol.reality_check(d, n_boot=200, block_length=2.0, rng=1)


def test_reality_check_detects_a_strong_strategy():
    assert ol.reality_check(planted(), n_boot=500, block_length=2.0, rng=0).pvalue < 0.01


def test_reality_check_accepts_noise():
    assert ol.reality_check(noise(), n_boot=500, block_length=2.0, rng=0).pvalue > 0.05


def test_automatic_block_length_is_reported():
    assert ol.reality_check(noise(), n_boot=50, rng=0).block_length >= 1.0


def test_supplied_indices_replace_the_sampler():
    d = noise()
    indices = ol.stationary_bootstrap_indices(400, 60, mean_block=2.0, rng=3)
    result = ol.reality_check(d, indices=indices)
    assert result.n_boot == 60
    assert math.isnan(result.block_length)
    assert ol.spa_test(d, indices=indices).n_boot == 60
    assert ol.romano_wolf_stepdown(d, indices=indices).rejected.size == 8


def test_spa_pvalues_are_ordered():
    result = ol.spa_test(noise(), n_boot=400, block_length=2.0, rng=0)
    assert result.pvalue_lower <= result.pvalue_consistent <= result.pvalue_upper


def test_spa_statistic_is_floored_at_zero():
    d = np.abs(noise()) * -1.0
    assert ol.spa_test(d, n_boot=50, block_length=2.0, rng=0).stat == 0.0


def test_unstudentised_statistic_has_no_floor():
    d = np.abs(noise()) * -1.0
    result = ol.spa_test(d, n_boot=50, block_length=2.0, rng=0, studentize=False)
    assert result.stat == pytest.approx(math.sqrt(400) * d.mean(axis=0).max())
    assert result.stat < 0.0


def test_spa_detects_a_strong_strategy():
    assert ol.spa_test(planted(), n_boot=500, block_length=2.0, rng=0).pvalue_consistent < 0.01


def test_spa_ignores_a_very_bad_strategy():
    # A hopeless strategy keeps the upper p-value high and leaves the consistent one alone.
    d = noise(1000, 4)
    d[:, 1] -= 0.02
    result = ol.spa_test(d, n_boot=400, block_length=2.0, rng=0)
    assert result.pvalue_consistent <= result.pvalue_upper


def test_studentising_favours_the_low_volatility_winner():
    d = noise(800, 2)
    d[:, 0] = d[:, 0] * 0.2 + 0.001
    d[:, 1] = d[:, 1] * 5.0
    studentised = ol.spa_test(d, n_boot=500, block_length=2.0, rng=0)
    raw = ol.spa_test(d, n_boot=500, block_length=2.0, rng=0, studentize=False)
    assert studentised.pvalue_consistent < raw.pvalue_consistent


def test_stepdown_finds_the_planted_strategy():
    result = ol.romano_wolf_stepdown(planted(), n_boot=500, block_length=2.0, rng=0)
    assert result.rejected.tolist() == [True] + [False] * 7
    assert result.adjusted_pvalues[0] <= 0.05
    assert result.adjusted_pvalues[1:].min() > 0.05


def test_stepdown_rejects_nothing_on_noise():
    result = ol.romano_wolf_stepdown(noise(), n_boot=500, block_length=2.0, rng=0)
    assert not result.rejected.any()
    assert result.n_steps == 1


def test_stepdown_takes_a_second_step():
    d = noise(1000, 20)
    d[:, 0] += 0.01
    d[:, 1] += 0.0028
    result = ol.romano_wolf_stepdown(d, n_boot=1000, block_length=2.0, rng=0)
    assert result.rejected[0]
    assert result.n_steps >= 2


def test_stepdown_rejects_everything_when_all_are_strong():
    d = noise(800, 3) + 0.02
    result = ol.romano_wolf_stepdown(d, n_boot=300, block_length=2.0, rng=0)
    assert result.rejected.all()


def test_adjusted_pvalues_keep_the_order_of_the_statistics():
    d = noise(600, 6) + np.array([0.004, 0.003, 0.002, 0.001, 0.0, -0.001])
    result = ol.romano_wolf_stepdown(d, n_boot=500, block_length=2.0, rng=0)
    tstats = d.mean(axis=0) / d.std(axis=0, ddof=1)
    assert (np.diff(result.adjusted_pvalues[np.argsort(-tstats)]) >= 0.0).all()


def test_stepdown_controls_the_familywise_error_rate():
    false_rejections = 0
    for seed in range(100):
        d = noise(200, 10, seed)
        result = ol.romano_wolf_stepdown(d, alpha=0.1, n_boot=200, block_length=1.0, rng=seed)
        false_rejections += int(result.rejected.any())
    assert false_rejections <= 18


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: ol.reality_check(noise(), n_boot=0), "n_boot"),
        (lambda: ol.reality_check(noise(), indices=np.zeros((2, 3), dtype=int)), "indices"),
        (lambda: ol.romano_wolf_stepdown(noise(), alpha=0.0), "alpha"),
    ],
)
def test_argument_validation(call, message):
    with pytest.raises(ValueError, match=message):
        call()
