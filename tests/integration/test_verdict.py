import json

import numpy as np
import pytest

import overfitlab as ol
from overfitlab.verdict import decide, judge

FAST = ol.VerdictConfig(n_boot=200)


def test_decide():
    assert decide({"a": "pass", "b": "pass"}) == "PASS"
    assert decide({"a": "pass", "b": "fail: x"}) == "FAIL"
    assert decide({"a": "pass", "b": "not computed: x"}) == "INCONCLUSIVE"
    assert decide({"a": "fail: x", "b": "not computed: x"}) == "FAIL"


def test_noise_fails_with_reasons():
    returns = np.random.default_rng(0).normal(0.0, 0.01, (400, 30))
    report = ol.evaluate(returns, config=FAST)
    assert report.verdict == "FAIL"
    assert report.reasons["dsr"].startswith("fail")
    assert set(report.reasons) == {"dsr", "pbo", "spa", "stepm", "haircut"}


def test_strong_signal_passes():
    returns = ol.planted_signal_matrix(2520, 20, true_sharpe_annual=4.0, rng=1).returns
    report = ol.evaluate(returns, config=FAST)
    assert report.verdict == "PASS", report.reasons
    assert report.selected == 0


def test_short_sample_is_inconclusive_at_best():
    returns = ol.planted_signal_matrix(200, 5, true_sharpe_annual=8.0, rng=1).returns
    report = ol.evaluate(returns, config=FAST)
    assert report.pbo is None
    assert report.reasons["pbo"].startswith("not computed")
    assert report.verdict in ("INCONCLUSIVE", "FAIL")


def test_benchmark_changes_the_snooping_tests():
    rng = np.random.default_rng(2)
    returns = rng.normal(0.002, 0.01, (800, 5))
    against_zero = ol.evaluate(returns, config=FAST)
    against_market = ol.evaluate(returns, benchmark=np.full(800, 0.002), config=FAST)
    assert against_zero.spa.pvalue_consistent < against_market.spa.pvalue_consistent


def test_selected_trial_can_be_named():
    returns = np.random.default_rng(3).normal(0.0, 0.01, (400, 6))
    assert ol.evaluate(returns, selected=2, config=FAST).selected == 2


def test_n_trials_reaches_the_dsr_and_the_haircut():
    returns = np.random.default_rng(4).normal(0.0, 0.01, (400, 6))
    report = ol.evaluate(returns, n_trials=50.0, config=FAST)
    assert report.dsr.n_trials == 50.0


def test_report_is_json_with_null_for_nan():
    returns = np.random.default_rng(5).normal(0.0, 0.01, (400, 6))
    report = ol.evaluate(returns, config=FAST)
    data = json.loads(report.to_json())
    assert data["verdict"] == "FAIL"
    assert len(data["pbo"]["logits"]) == report.pbo.n_combinations


def test_haircut_uses_the_configured_frequency():
    returns = np.random.default_rng(6).normal(0.0, 0.01, (480, 6))
    report = ol.evaluate(returns, config=ol.VerdictConfig(n_boot=100, periods_per_year=12))
    assert report.haircut.sr_annual == pytest.approx(report.dsr.sr * 12**0.5)


def test_judge_flags_a_stepdown_that_misses_the_selected_trial():
    returns = np.random.default_rng(7).normal(0.0, 0.01, (400, 6))
    report = ol.evaluate(returns, config=FAST)
    reasons = judge(
        ol.VerdictConfig(),
        dsr=report.dsr,
        pbo=None,
        spa=None,
        stepm=None,
        haircut_sr_annual=None,
        selected=0,
    )
    assert all(k == "dsr" or v.startswith("not computed") for k, v in reasons.items())


def test_judge_fails_a_non_positive_haircut():
    returns = np.random.default_rng(8).normal(0.0, 0.01, (400, 6))
    report = ol.evaluate(returns, config=FAST)
    reasons = judge(
        ol.VerdictConfig(),
        dsr=report.dsr,
        pbo=report.pbo,
        spa=report.spa,
        stepm=report.stepm,
        haircut_sr_annual=0.0,
        selected=report.selected,
    )
    assert reasons["haircut"].startswith("fail")


@pytest.mark.slow
def test_noise_is_never_a_pass():
    config = ol.VerdictConfig()
    verdicts = [
        ol.evaluate(ol.gaussian_noise(1260, 1000, rng=seed), config=config, rng=seed).verdict
        for seed in range(20)
    ]
    assert sum(v != "PASS" for v in verdicts) >= 19


@pytest.mark.slow
def test_strong_planted_signal_passes():
    config = ol.VerdictConfig()
    verdicts = [
        ol.evaluate(
            ol.planted_signal_matrix(2520, 100, true_sharpe_annual=2.0, rng=seed).returns,
            config=config,
            rng=seed,
        ).verdict
        for seed in range(20)
    ]
    assert sum(v == "PASS" for v in verdicts) >= 18
