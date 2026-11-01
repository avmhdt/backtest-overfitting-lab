import numpy as np
import pytest
from typer.testing import CliRunner

from overfitlab_study.cli import app
from overfitlab_study.config import experiment_seeds, load_study_config
from overfitlab_study.reports import format_cell
from overfitlab_study.sims import leakage

FAST = leakage.LeakageConfig(
    n_obs=300,
    label_horizon=10,
    n_features=2,
    feature_ar1=0.9,
    n_splits=3,
    embargo_pct=0.01,
    n_estimators=5,
    max_depth=3,
    min_samples_leaf=5,
)


def accuracy_of(values):
    return {scheme: np.array(values) for scheme in leakage.SCHEMES}


def test_config_from_the_study_file():
    config = leakage.LeakageConfig.from_mapping(load_study_config()["leakage"])
    assert (config.n_obs, config.label_horizon, config.n_estimators) == (2000, 20, 200)


def test_dataset_shapes_and_overlapping_labels():
    features, labels, t1 = leakage.make_dataset(FAST, np.random.default_rng(0))
    assert features.shape == (300, 2)
    assert set(np.unique(labels)) == {0, 1}
    assert t1[0] == 10
    # Neighbouring labels share all but one return, so they mostly agree.
    assert (labels[1:] == labels[:-1]).mean() > 0.7


def test_seed_outcome_is_reproducible():
    seed = experiment_seeds(1, "test", 1)[0]
    first = leakage.run_seed(FAST, seed)
    second = leakage.run_seed(FAST, experiment_seeds(1, "test", 1)[0])
    assert first == second
    assert set(first) == set(leakage.SCHEMES)


def test_seeds_differ_by_experiment_and_replication():
    a, b = experiment_seeds(1, "x", 2)
    c = experiment_seeds(1, "y", 1)[0]
    assert a.generate_state(1)[0] != b.generate_state(1)[0]
    assert a.generate_state(1)[0] != c.generate_state(1)[0]


def test_summary_rows():
    rows = leakage.summarise(accuracy_of([0.5, 0.52, 0.48]))
    assert [row["row"] for row in rows] == [
        *leakage.SCHEMES,
        "kfold_shuffled_minus_purged",
        "kfold_unshuffled_minus_purged",
    ]
    assert rows[0]["mean"] == pytest.approx(0.5)
    assert rows[0]["ci_low"] < 0.5 < rows[0]["ci_high"]


def test_criteria_pass_on_a_clear_gap():
    accuracy = accuracy_of([0.49, 0.5, 0.51])
    accuracy["kfold_shuffled"] = np.array([0.6, 0.61, 0.59])
    assert all(leakage.check_criteria(leakage.summarise(accuracy)).values())


def test_criteria_fail_without_a_gap():
    criteria = leakage.check_criteria(leakage.summarise(accuracy_of([0.49, 0.5, 0.51])))
    assert not criteria["gap_at_least_min"]


def test_cell_format():
    assert format_cell(1.23456789) == "1.23457"
    assert format_cell(True) == "true"
    assert format_cell(None) == ""
    assert format_cell(5) == "5"


def test_run_writes_table_figure_and_raw_results(tmp_path):
    rows, criteria = leakage.run(
        FAST,
        seeds=2,
        master_seed=1,
        reports_dir=tmp_path / "reports",
        results_dir=tmp_path / "results",
        config_key="demo",
    )
    assert len(rows) == 6
    assert set(criteria) == {"gap_at_least_min", "gap_ci_excludes_zero", "purged_ci_contains_half"}
    assert (tmp_path / "reports/tables/leakage.csv").exists()
    assert (tmp_path / "reports/tables/leakage.md").exists()
    assert (tmp_path / "reports/figures/leakage_kfold_vs_purged.png").stat().st_size > 1000
    assert (tmp_path / "results/sims/leakage/demo.parquet").exists()


@pytest.mark.slow
def test_leakage_quick_shows_the_gap(tmp_path, monkeypatch):
    monkeypatch.setenv("OLAB_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("OLAB_REPORTS_DIR", str(tmp_path / "reports"))
    result = CliRunner().invoke(app, ["sim", "leakage", "--seeds", "4"])
    assert result.exit_code == 0, result.output
    assert "gap_at_least_min': True" in result.stdout
