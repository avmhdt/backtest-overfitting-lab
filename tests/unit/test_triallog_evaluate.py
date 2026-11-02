import numpy as np
import polars as pl
import pytest

import overfitlab as ol
from overfitlab import TrialLogger, triallog


def days(count):
    return np.datetime64("2020-01-01", "D") + np.arange(count)


@pytest.fixture
def logger(tmp_path):
    return TrialLogger(tmp_path / "trial_log.csv", tmp_path / "runs", repo_dir=tmp_path)


def fill(logger, matrix, *, family="demo", extra=None):
    return [
        logger.log(
            strategy_family=family,
            params={"n": n},
            dates=days(matrix.shape[0]),
            returns=matrix[:, n],
            extra=extra,
        )
        for n in range(matrix.shape[1])
    ]


def clustered_returns(n_clusters, per_cluster, n_obs=600, seed=0):
    rng = np.random.default_rng(seed)
    factors = rng.normal(0.0, 0.01, (n_obs, n_clusters))
    columns = [
        factors[:, c] + rng.normal(0.0, 0.001, n_obs)
        for c in range(n_clusters)
        for _ in range(per_cluster)
    ]
    return np.column_stack(columns)


def test_effective_trials_recover_the_clusters():
    assert triallog._effective_trials(clustered_returns(4, 5)) == 4


def test_effective_trials_treat_independent_series_as_independent():
    returns = np.random.default_rng(0).normal(0.0, 0.01, (600, 12))
    assert triallog._effective_trials(returns) == 12


def test_effective_trials_of_two_series():
    assert triallog._effective_trials(np.random.default_rng(0).normal(size=(50, 2))) == 2


def test_silhouette_by_hand():
    distances = np.array([[0.0, 0.1, 1.0], [0.1, 0.0, 1.0], [1.0, 1.0, 0.0]])
    labels = np.array([1, 1, 2])
    # Both members of the pair score 1 - 0.1, and the singleton scores 0.
    assert triallog._mean_silhouette(distances, labels) == pytest.approx(0.6)


def test_clustered_count_adds_rows_without_a_series(logger, tmp_path):
    fill(logger, clustered_returns(3, 4))
    logger.log_failed(strategy_family="demo", params={"n": 99}, note="crashed")
    log = ol.read_trial_log(tmp_path / "trial_log.csv")
    assert log.n_trials(mode="clustered") == 4
    assert log.n_trials() == 13


def test_clustered_count_needs_common_observations(logger, tmp_path):
    fill(logger, clustered_returns(2, 2, n_obs=100))
    with pytest.raises(ValueError, match="252"):
        ol.read_trial_log(tmp_path / "trial_log.csv").n_trials(mode="clustered")


def test_clustered_n_reaches_the_report(logger, tmp_path):
    fill(logger, clustered_returns(3, 4))
    report = ol.evaluate_trial_log(tmp_path / "trial_log.csv", n_effective="clustered")
    assert report.n_trials == 3
    assert report.n_trials_mode == "clustered"


def test_report_carries_the_verdict_on_noise(logger, tmp_path):
    matrix = np.random.default_rng(0).normal(0.0, 0.01, (600, 8))
    fill(logger, matrix)
    report = ol.evaluate_trial_log(tmp_path / "trial_log.csv")
    assert report.verdict == "FAIL"
    assert report.pbo is not None
    assert report.spa_pvalue is not None
    assert report.stepm_rejects_selected is False


def test_report_passes_on_a_strong_signal(logger, tmp_path):
    returns = ol.planted_signal_matrix(1500, 10, true_sharpe_annual=5.0, rng=1).returns
    fill(logger, returns)
    report = ol.evaluate_trial_log(tmp_path / "trial_log.csv")
    assert report.verdict == "PASS", report.reasons
    assert report.stepm_rejects_selected is True


def test_short_logs_are_at_most_inconclusive(logger, tmp_path):
    returns = ol.planted_signal_matrix(200, 5, true_sharpe_annual=9.0, rng=1).returns
    fill(logger, returns)
    report = ol.evaluate_trial_log(tmp_path / "trial_log.csv")
    assert report.pbo is None
    assert report.reasons["pbo"].startswith("not computed")
    assert report.verdict in ("INCONCLUSIVE", "FAIL")


def test_summary_only_logs_are_at_most_inconclusive(logger, tmp_path):
    returns = ol.planted_signal_matrix(600, 5, true_sharpe_annual=9.0, rng=1).returns
    fill(logger, returns)
    for path in (tmp_path / "runs").rglob("*.parquet"):
        path.unlink()
    report = ol.evaluate_trial_log(tmp_path / "trial_log.csv")
    assert report.spa_pvalue is None
    assert report.verdict in ("INCONCLUSIVE", "FAIL")


def test_named_benchmark_is_subtracted(logger, tmp_path):
    rng = np.random.default_rng(3)
    market = rng.normal(0.003, 0.01, 800)
    benchmark = logger.log(
        strategy_family="market",
        params={},
        dates=days(800),
        returns=market,
        extra={"x_benchmark_run_id": ""},
    )
    returns = market[:, np.newaxis] + rng.normal(0.0, 0.002, (800, 6))
    fill(logger, returns, extra={"x_benchmark_run_id": benchmark}, family="demo")
    path = tmp_path / "trial_log.csv"
    assert "x_benchmark_run_id" in path.read_text("utf-8").splitlines()[0]
    with_benchmark = ol.evaluate_trial_log(path, families=["demo"])
    assert with_benchmark.spa_pvalue > 0.05


def test_inconsistent_benchmarks_are_rejected(tmp_path):
    rows = [{"x_benchmark_run_id": "a"}, {"x_benchmark_run_id": "b"}]
    log = ol.TrialLog(rows, tmp_path)
    with pytest.raises(ValueError, match="more than one"):
        triallog._benchmark_series(log, days(3))


def test_missing_benchmark_is_rejected(tmp_path):
    log = ol.TrialLog([{"x_benchmark_run_id": "a"}], tmp_path)
    with pytest.raises(ValueError, match="missing"):
        triallog._benchmark_series(log, days(3))


def test_short_benchmark_is_rejected(logger, tmp_path):
    rng = np.random.default_rng(3)
    run_id = logger.log(
        strategy_family="market", params={}, dates=days(10), returns=rng.normal(0, 0.01, 10)
    )
    log = ol.TrialLog([{"x_benchmark_run_id": run_id}], tmp_path)
    with pytest.raises(ValueError, match="does not cover"):
        triallog._benchmark_series(log, days(20))


def test_read_wide_csv(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text("date,a,b\n2024-01-01,1,2\n2024-01-02,3,4\n", encoding="utf-8")
    names, matrix = ol.triallog.read_wide_returns(path)
    assert names == ("a", "b")
    assert matrix.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_read_wide_parquet(tmp_path):
    path = tmp_path / "wide.parquet"
    pl.DataFrame({"date": ["2024-01-01"], "a": [0.5], "b": [1.5]}).write_parquet(path)
    names, matrix = ol.triallog.read_wide_returns(path)
    assert names == ("a", "b")
    assert matrix.tolist() == [[0.5, 1.5]]


def test_read_wide_needs_trial_columns(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text("date\n2024-01-01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no trial columns"):
        ol.triallog.read_wide_returns(path)
