import numpy as np
import pytest

import overfitlab as ol


def test_logger_validate_evaluate_round_trip(tmp_path):
    trials = ol.TrialLogger(tmp_path / "trial_log.csv", tmp_path / "runs", repo_dir=tmp_path)
    days = np.datetime64("2023-01-02", "D") + np.arange(500)
    rng = np.random.default_rng(0)
    matrix = rng.normal(0.0, 0.01, (days.size, 8))
    matrix[:, 3] += 0.002
    for n in range(matrix.shape[1]):
        trials.log(strategy_family="demo", params={"n": n}, dates=days, returns=matrix[:, n])
    trials.log_failed(strategy_family="demo", params={"n": 99}, note="aborted")

    log_path = tmp_path / "trial_log.csv"
    assert [i.code for i in ol.validate_trial_log(log_path) if i.severity == "error"] == []

    report = ol.evaluate_trial_log(log_path)
    direct = ol.dsr_from_returns(matrix, n_trials=9.0)
    assert report.n_trials == 9
    assert report.n_valid == 8
    assert report.dsr == pytest.approx(direct.dsr)
    assert report.sr0 == pytest.approx(direct.sr0)
    assert report.haircut_sr_annual is not None

    _, aligned, run_ids = ol.read_trial_log(log_path).returns_matrix()
    assert aligned.shape == matrix.shape
    assert np.array_equal(aligned, matrix)
    assert len(run_ids) == 8
