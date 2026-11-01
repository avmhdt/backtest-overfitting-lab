import numpy as np
import pytest

import overfitlab as ol


def block_lengths(indices):
    """Lengths of runs of consecutive indices, wrapping at the end of the sample."""
    n_obs = indices.shape[1]
    continues = indices[:, 1:] == (indices[:, :-1] + 1) % n_obs
    lengths = []
    for row in continues:
        breaks = np.flatnonzero(~row)
        lengths.extend(np.diff(breaks).tolist())
    return np.array(lengths)


def test_indices_shape_and_range():
    indices = ol.stationary_bootstrap_indices(50, 7, mean_block=4.0, rng=0)
    assert indices.shape == (7, 50)
    assert indices.dtype == np.int64
    assert indices.min() >= 0
    assert indices.max() < 50


def test_indices_are_reproducible_and_seeded():
    first = ol.stationary_bootstrap_indices(50, 3, mean_block=4.0, rng=1)
    second = ol.stationary_bootstrap_indices(50, 3, mean_block=4.0, rng=1)
    other = ol.stationary_bootstrap_indices(50, 3, mean_block=4.0, rng=2)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_mean_block_of_one_is_iid():
    indices = ol.stationary_bootstrap_indices(1000, 1, mean_block=1.0, rng=0)
    assert ((indices[:, 1:] == (indices[:, :-1] + 1) % 1000).mean()) < 0.01


@pytest.mark.parametrize("mean_block", [2.0, 5.0, 20.0])
def test_block_lengths_have_the_requested_mean(mean_block):
    indices = ol.stationary_bootstrap_indices(2000, 100, mean_block=mean_block, rng=0)
    assert block_lengths(indices).mean() == pytest.approx(mean_block, rel=0.1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_obs": 0, "n_boot": 1, "mean_block": 2.0}, "n_obs"),
        ({"n_obs": 5, "n_boot": 0, "mean_block": 2.0}, "n_boot"),
        ({"n_obs": 5, "n_boot": 1, "mean_block": 0.5}, "mean_block"),
    ],
)
def test_indices_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ol.stationary_bootstrap_indices(kwargs.pop("n_obs"), kwargs.pop("n_boot"), **kwargs)


def ar1(n_obs, phi, seed):
    rng = np.random.default_rng(seed)
    series = np.zeros(n_obs)
    for t in range(1, n_obs):
        series[t] = phi * series[t - 1] + rng.standard_normal()
    return series


def test_block_length_is_short_on_white_noise():
    assert ol.optimal_block_length(np.random.default_rng(0).standard_normal(5000)) < 5


def test_block_length_grows_with_persistence():
    lengths = [ol.optimal_block_length(ar1(5000, phi, 0)) for phi in (0.0, 0.5, 0.9)]
    assert lengths[0] < lengths[1] < lengths[2]


def test_block_length_of_constant_series_is_one():
    assert ol.optimal_block_length(np.ones(50)) == 1.0


def test_block_length_respects_the_cap():
    assert ol.optimal_block_length(np.cumsum(np.ones(100))) <= 3 * np.sqrt(100)


def test_block_length_needs_observations():
    with pytest.raises(ValueError, match="9 observations"):
        ol.optimal_block_length(np.arange(5.0))
