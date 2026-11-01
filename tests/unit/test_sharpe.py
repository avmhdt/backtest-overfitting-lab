import math

import numpy as np
import polars as pl
import pytest
from scipy import stats

import overfitlab as ol


def test_sharpe_matches_definition():
    r = np.array([0.01, -0.02, 0.03, 0.005])
    assert ol.sharpe_ratio(r) == pytest.approx(r.mean() / r.std(ddof=1))


def test_sharpe_of_matrix_is_per_column():
    r = np.random.default_rng(0).normal(size=(50, 3))
    expected = r.mean(axis=0) / r.std(axis=0, ddof=1)
    assert ol.sharpe_ratio(r) == pytest.approx(expected)


def test_sharpe_along_rows():
    r = np.random.default_rng(0).normal(size=(3, 50))
    assert ol.sharpe_ratio(r, axis=1) == pytest.approx(ol.sharpe_ratio(r.T))


def test_flat_series_has_zero_sharpe():
    assert ol.sharpe_ratio([0.0, 0.0, 0.0]) == 0.0
    assert ol.sharpe_ratio([0.01, 0.01, 0.01]) == 0.0


def test_flat_column_does_not_break_a_matrix():
    r = np.column_stack((np.zeros(10), np.arange(10.0)))
    out = ol.sharpe_ratio(r)
    assert out[0] == 0.0
    assert out[1] > 0.0


def test_sharpe_needs_two_observations():
    with pytest.raises(ValueError, match="n_obs"):
        ol.sharpe_ratio([0.01])


def test_sharpe_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        ol.sharpe_ratio([0.01, np.nan])


def test_polars_parity():
    values = [0.01, -0.02, 0.03]
    assert ol.sharpe_ratio(pl.Series(values)) == ol.sharpe_ratio(values)
    assert ol.return_moments(pl.Series(values)) == ol.return_moments(values)


def test_moments_match_scipy():
    r = np.random.default_rng(1).standard_t(5, 400)
    m = ol.return_moments(r)
    assert m.skew == pytest.approx(stats.skew(r))
    assert m.kurtosis == pytest.approx(stats.kurtosis(r, fisher=False))
    assert m.n_obs == 400


def test_moments_reject_constant_series():
    with pytest.raises(ValueError, match="zero variance"):
        ol.return_moments([0.01, 0.01])


def test_moments_need_two_observations():
    with pytest.raises(ValueError, match="n_obs"):
        ol.return_moments([0.01])


def test_annualize_iid():
    assert ol.annualize_sharpe(0.1, 252) == pytest.approx(0.1 * math.sqrt(252))
    assert ol.annualize_sharpe(0.1, 252, autocorr=0.0) == pytest.approx(0.1 * math.sqrt(252))


def test_annualize_matches_direct_sum():
    # Lo (2002) eq. 18 written as a sum: q / sqrt(q + 2 * sum((q - k) * rho^k)).
    q, rho = 12, 0.3
    direct = q / math.sqrt(q + 2 * sum((q - k) * rho**k for k in range(1, q)))
    assert ol.annualize_sharpe(1.0, q, autocorr=rho) == pytest.approx(direct)


@pytest.mark.parametrize("rho", [-1.0, 1.0, 1.5])
def test_annualize_rejects_bad_autocorr(rho):
    with pytest.raises(ValueError, match="autocorr"):
        ol.annualize_sharpe(1.0, 12, autocorr=rho)


def test_annualize_rejects_bad_periods():
    with pytest.raises(ValueError, match="periods_per_year"):
        ol.annualize_sharpe(1.0, 0)


def test_sharpe_std_normal_case():
    assert ol.sharpe_std(0.5, 101) == pytest.approx(math.sqrt((1 + 0.125) / 100))


def test_psr_at_benchmark_is_one_half():
    assert ol.probabilistic_sharpe_ratio(0.2, 100, sr_benchmark=0.2) == pytest.approx(0.5)


def test_psr_stable_far_in_the_tail():
    assert ol.probabilistic_sharpe_ratio(-5.0, 10_000) < 1e-12


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_obs": 1}, "n_obs"),
        ({"n_obs": 100, "kurtosis": 0.5}, "kurtosis"),
        ({"n_obs": 100, "skew": 10.0}, "variance term"),
    ],
)
def test_psr_argument_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ol.probabilistic_sharpe_ratio(0.5, **kwargs)


def test_psr_from_returns_uses_own_moments():
    r = np.random.default_rng(2).normal(0.001, 0.01, 500)
    m = ol.return_moments(r)
    assert ol.psr_from_returns(r) == pytest.approx(
        ol.probabilistic_sharpe_ratio(m.sr, m.n_obs, skew=m.skew, kurtosis=m.kurtosis)
    )


def test_mintrl_is_where_psr_reaches_the_confidence_level():
    sr, benchmark = 0.2, 0.05
    length = ol.min_track_record_length(sr, sr_benchmark=benchmark, skew=-0.5, kurtosis=5.0)
    psr = ol.probabilistic_sharpe_ratio(
        sr, math.ceil(length), sr_benchmark=benchmark, skew=-0.5, kurtosis=5.0
    )
    assert psr >= 0.95


def test_mintrl_rejects_sr_below_benchmark():
    with pytest.raises(ValueError, match="exceed"):
        ol.min_track_record_length(0.1, sr_benchmark=0.2)


def test_mintrl_rejects_bad_alpha():
    with pytest.raises(ValueError, match="alpha"):
        ol.min_track_record_length(0.1, alpha=1.5)
