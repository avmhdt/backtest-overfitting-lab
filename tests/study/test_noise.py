import numpy as np
import pytest
from typer.testing import CliRunner

from overfitlab_study.cli import app
from overfitlab_study.config import load_study_config
from overfitlab_study.sims import noise

SMALL = noise.NoiseConfig(
    seeds=2,
    n_obs=360,
    vol_annual=0.16,
    n_strategies=12,
    pbo_partitions=8,
    n_boot=50,
    cost_bps=1.0,
    quick=True,
)


def outcome(setting, **changes):
    base = {
        "setting": setting,
        "n_trials": 1000,
        "best_sr_annual": 1.5,
        "naive_psr": 0.99,
        "dsr": 0.5,
        "pbo": 0.5,
        "rc_pvalue": 0.5,
        "spa_pvalue": 0.5,
        "verdict": "FAIL",
    }
    return {**base, **changes}


def test_config_from_the_study_file():
    study = load_study_config()
    config = noise.NoiseConfig.from_mapping(study["noise"], study["factory"]["cost_bps"])
    assert (config.seeds, config.n_obs, config.n_strategies) == (100, 1260, 1000)


def test_quick_config_is_smaller():
    study = load_study_config()
    config = noise.NoiseConfig.from_mapping(study["noise"], 1.0, quick=True)
    assert (config.seeds, config.n_strategies) == (20, 200)


def test_independent_returns_have_the_requested_shape():
    returns = noise.strategy_returns("independent", SMALL, np.random.default_rng(0))
    assert returns.shape == (360, 12)


def test_factory_returns_use_the_grid():
    returns = noise.strategy_returns("factory_gbm", SMALL, np.random.default_rng(0))
    assert returns.shape[0] == 360
    assert 150 <= returns.shape[1] <= 300


def test_summary_counts():
    rows = noise.summarise(
        [
            outcome("independent"),
            outcome("independent", dsr=0.97, verdict="PASS"),
            outcome("factory_gbm"),
        ]
    )
    assert rows[0]["seeds_dsr_below_95"] == 1
    assert rows[0]["seeds_pass"] == 1
    assert rows[1]["n_seeds"] == 1


def test_criteria_pass_on_the_expected_picture():
    rows = noise.summarise([outcome("independent"), outcome("factory_gbm")])
    assert all(noise.check_criteria(rows).values())


def test_criteria_catch_a_significant_dsr():
    rows = noise.summarise([outcome("independent", dsr=0.99), outcome("factory_gbm")])
    assert not noise.check_criteria(rows)["dsr_not_significant_independent"]


def test_run_writes_outputs(tmp_path):
    rows, criteria = noise.run(
        SMALL,
        seeds=2,
        master_seed=1,
        reports_dir=tmp_path / "reports",
        results_dir=tmp_path / "results",
        config_key="demo",
    )
    assert [row["setting"] for row in rows] == list(noise.SETTINGS)
    assert len(criteria) == 6
    assert (tmp_path / "reports/tables/noise_experiment.csv").exists()
    assert (tmp_path / "reports/figures/noise_best_vs_dsr.png").stat().st_size > 1000
    assert (tmp_path / "results/sims/noise/demo.parquet").exists()


@pytest.mark.slow
def test_noise_quick(tmp_path, monkeypatch):
    monkeypatch.setenv("OLAB_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("OLAB_REPORTS_DIR", str(tmp_path / "reports"))
    result = CliRunner().invoke(app, ["sim", "noise", "--quick"])
    assert result.exit_code == 0, result.output
    assert "False" not in result.stdout.split("AC-W-01:")[1]
