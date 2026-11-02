import math

import numpy as np
import pytest
from scipy import stats

import overfitlab as ol
from overfitlab.haircut import _adjusted_for_last, _hlz_parameters


def test_tstat_definition():
    assert ol.sharpe_to_tstat(1.0, 120, 12) == pytest.approx(math.sqrt(10))


def test_tstat_uses_whole_months():
    assert ol.sharpe_to_tstat(1.0, 2520, 252) == pytest.approx(ol.sharpe_to_tstat(1.0, 120, 12))


def test_autocorr_reduces_the_tstat():
    assert ol.sharpe_to_tstat(1.0, 120, 12, autocorr=0.1) < ol.sharpe_to_tstat(1.0, 120, 12)


def test_short_sample_is_rejected():
    with pytest.raises(ValueError, match="two months"):
        ol.haircut_sharpe(1.0, 10, periods_per_year=252, n_tests=10)


def test_one_test_leaves_bonferroni_unchanged():
    result = ol.haircut_sharpe(1.0, 120, periods_per_year=12, n_tests=1, method="bonferroni")
    assert result.haircut_sr_annual == pytest.approx(1.0)
    assert result.haircut_pct == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(("sr_annual", "years"), [(3.0, 20), (5.0, 20), (12.0, 50), (40.0, 50)])
def test_strong_sharpe_ratio_keeps_a_finite_haircut(sr_annual, years):
    result = ol.haircut_sharpe(
        sr_annual, years * 12, periods_per_year=12, n_tests=100, method="bonferroni"
    )
    assert 0.0 < result.haircut_sr_annual <= sr_annual
    assert 0.0 <= result.haircut_pct < 100.0


def test_pvalue_is_two_sided_student_t():
    result = ol.haircut_sharpe(1.0, 120, periods_per_year=12, n_tests=5, method="bonferroni")
    assert result.pvalue == pytest.approx(2 * stats.t.sf(result.tstat, 119))


def test_bonferroni_saturates_at_one():
    result = ol.haircut_sharpe(0.1, 120, periods_per_year=12, n_tests=100, method="bonferroni")
    assert result.adj_pvalue == 1.0
    assert result.haircut_sr_annual == pytest.approx(0.0, abs=1e-12)


def test_negative_sharpe_has_nothing_to_deflate():
    result = ol.haircut_sharpe(-0.5, 120, periods_per_year=12, n_tests=10)
    assert (result.adj_pvalue, result.haircut_sr_annual, result.haircut_pct) == (1.0, 0.0, 100.0)


@pytest.mark.parametrize("method", ["holm", "bhy"])
def test_simulation_is_seeded(method):
    args = {"periods_per_year": 12, "n_tests": 20, "method": method, "n_sims": 300}
    assert ol.haircut_sharpe(1.0, 120, **args, rng=7) == ol.haircut_sharpe(1.0, 120, **args, rng=7)


def test_methods_rank_as_in_the_paper():
    args = {"periods_per_year": 12, "n_tests": 100, "autocorr": 0.1, "rho_tests": 0.4}
    bonf, holm, bhy = (
        ol.haircut_sharpe(1.0, 120, method=m, n_sims=1000, **args).haircut_sr_annual
        for m in ("bonferroni", "holm", "bhy")
    )
    assert bonf < holm < bhy


def test_large_test_count_runs_in_chunks():
    result = ol.haircut_sharpe(2.0, 240, periods_per_year=12, n_tests=2000, n_sims=600)
    assert 0.0 < result.haircut_sr_annual < 2.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_tests": 0}, "n_tests"),
        ({"n_tests": 5, "method": "sidak"}, "method"),
        ({"n_tests": 5, "n_sims": 0}, "n_sims"),
        ({"n_tests": 5, "rho_tests": 1.0}, "rho_tests"),
    ],
)
def test_argument_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ol.haircut_sharpe(1.0, 120, periods_per_year=12, **kwargs)


def test_parameters_at_the_grid():
    assert _hlz_parameters(0.2) == pytest.approx((0.44589, 0.0055508))


def test_parameters_interpolate_as_in_the_paper():
    # The appendix's example at rho = 0.3 averages the 0.2 and 0.4 rows.
    assert _hlz_parameters(0.3)[0] == pytest.approx((0.44589 + 0.48604) / 2)


def test_parameters_extrapolate_above_the_grid():
    assert _hlz_parameters(0.9)[0] > 0.83901


def test_holm_by_hand():
    # Sorted 0.01, 0.03, 0.04 with multipliers 3, 2, 1 gives 0.03, 0.06, 0.06.
    p = np.array([[0.04, 0.01, 0.03]])
    assert _adjusted_for_last(p, "holm") == pytest.approx([0.06])


def test_bhy_by_hand():
    # c = 1 + 1/2 + 1/3 = 11/6, so factors are 5.5, 2.75, unscaled for the largest.
    p = np.array([[0.04, 0.01, 0.02]])
    assert _adjusted_for_last(p, "bhy") == pytest.approx([0.04])
    q = np.array([[0.04, 0.02, 0.01]])
    assert _adjusted_for_last(q, "bhy") == pytest.approx([min(5.5 * 0.01, 2.75 * 0.02, 0.04)])
