import math

import numpy as np
import pytest
from typer.testing import CliRunner

from overfitlab_study.cli import app
from overfitlab_study.config import load_study_config
from overfitlab_study.settings import Settings
from overfitlab_study.sims import power, theory


@pytest.fixture
def config():
    return power.PowerConfig.from_mapping(load_study_config(), quick=True)


def rows_for(powers, test="dsr", years=5.0, n_trials=10):
    return [
        {
            "test": test,
            "years": years,
            "n_trials": n_trials,
            "true_sharpe_annual": sharpe,
            "power": value,
        }
        for sharpe, value in powers.items()
    ]


def test_psr_power_at_zero_signal_is_alpha():
    assert theory.psr_power(0.0, 1260, 0.05) == pytest.approx(0.05)


def test_max_stat_power_at_zero_signal_is_alpha():
    c = theory.critical_value("rc", 100, 0.05)
    assert theory.max_stat_power(0.0, 1260, 100, c) == pytest.approx(0.05)


def test_power_rises_with_signal_and_length():
    low = theory.reference_power(
        "rc", true_sharpe_annual=0.5, years=3, n_trials=10, alpha=0.05, periods_per_year=252
    )
    high = theory.reference_power(
        "rc", true_sharpe_annual=1.5, years=10, n_trials=10, alpha=0.05, periods_per_year=252
    )
    assert low < high


def test_dsr_critical_value_with_one_trial_is_the_normal_quantile():
    assert theory.critical_value("dsr", 1, 0.05) == pytest.approx(1.6448536, abs=1e-6)


def test_expected_max_normal_matches_simulation():
    draws = np.random.default_rng(0).standard_normal((20_000, 100)).max(axis=1)
    assert theory.expected_max_normal(100) == pytest.approx(draws.mean(), abs=0.03)


def test_no_reference_for_other_tests():
    with pytest.raises(ValueError, match="pbo"):
        theory.reference_power(
            "pbo", true_sharpe_annual=1.0, years=1, n_trials=2, alpha=0.05, periods_per_year=252
        )


def test_config_quick_and_full():
    study = load_study_config()
    full = power.PowerConfig.from_mapping(study)
    assert (full.reps_closed_form, full.reps_bootstrap, full.n_boot) == (1000, 500, 1000)
    assert len(full.grid()) == 7 * 6 * 3
    assert len(full.check_cells) == 4


def test_closed_rep_flags(config):
    outcome = power.closed_rep(power.Cell(3.0, 5.0, 2), config, np.random.SeedSequence(0))
    assert outcome == {"psr": 1.0, "dsr": 1.0, "dsr_known_v": 1.0}


def test_boot_rep_with_one_trial_has_no_pbo(config):
    outcome = power.boot_rep(power.Cell(1.0, 1.0, 1), config, np.random.SeedSequence(0))
    assert math.isnan(outcome["pbo"])


def test_boot_rep_detects_a_strong_signal(config):
    outcome = power.boot_rep(power.Cell(4.0, 3.0, 5), config, np.random.SeedSequence(1))
    assert outcome["stepm"] == 1.0
    assert outcome["pbo_value"] < 0.2


def test_rejection_rate_ignores_nan():
    assert power.rejection_rate(np.array([1.0, 0.0, np.nan])) == 0.5
    assert math.isnan(power.rejection_rate(np.array([np.nan])))


def test_min_length_picks_the_shortest_reaching_target():
    rows = [
        *rows_for({1.0: 0.5}, years=1.0),
        *rows_for({1.0: 0.85}, years=3.0),
        *rows_for({1.0: 0.95}, years=5.0),
    ]
    found = [r for r in power.min_length_rows(rows) if r["true_sharpe_annual"] == 1.0]
    assert found[0]["min_years_for_80pct"] == 3.0


def test_min_length_is_blank_when_never_reached():
    found = power.min_length_rows(rows_for({0.5: 0.2}))
    assert found[0]["min_years_for_80pct"] is None


def test_monotone_allows_monte_carlo_error():
    assert power.monotone_violations(rows_for({0.0: 0.1, 1.0: 0.5, 2.0: 0.48})) == []


def test_monotone_flags_a_real_fall():
    assert power.monotone_violations(rows_for({0.0: 0.1, 1.0: 0.5, 2.0: 0.3}))


def test_simulate_is_reproducible_and_covers_each_cell(config):
    small = power.PowerConfig(
        alpha=0.05,
        reps_closed_form=30,
        reps_bootstrap=5,
        n_boot=50,
        corr=0.0,
        true_sharpes=(1.0,),
        years=(1.0,),
        n_trials=(2,),
        check_cells=(power.Cell(1.0, 1.0, 2),),
        tests=config.tests,
    )
    cell = power.Cell(1.0, 1.0, 2)
    first = power.simulate([cell], small, 1)
    second = power.simulate([cell], small, 1)
    assert first[cell]["psr"].size == 30
    assert first[cell]["rc"].size == 5
    assert np.array_equal(first[cell]["dsr"], second[cell]["dsr"])
    rows = power.grid_rows(first, small)
    assert {row["test"] for row in rows} == set(power.TEST_ORDER)


def test_check_rows_mark_the_tolerance(config):
    cell = power.Cell(1.0, 3, 1)
    tests = [*power.TEST_ORDER, "dsr_known_v"]
    results = {c: {t: np.array([0.0]) for t in tests} for c in config.check_cells}
    results[cell]["psr"] = np.full(100, 0.8)
    rows = power.check_rows(results, config, quick=False)
    row = next(r for r in rows if r["test"] == "psr")
    assert row["tolerance"] == 0.03
    assert not row["within_tolerance"]
    assert [r["diagnostic"] for r in rows if r["test"] == "dsr_known_v"] == [True] * 4


def test_plots_are_written(tmp_path):
    rows = []
    for test in ("dsr", "rc"):
        for n_trials in (10, 100):
            for years in (1.0, 5.0):
                rows += rows_for({0.0: 0.05, 1.0: 0.5, 2.0: 0.9}, test, years, n_trials)
    power.plot(rows, tmp_path / "figures" / "power.png", ["dsr", "rc"])
    power.plot_html(rows, tmp_path / "docs" / "power.html", ["dsr", "rc"])
    assert (tmp_path / "figures" / "power.png").stat().st_size > 1000
    assert "plotly" in (tmp_path / "docs" / "power.html").read_text("utf-8").lower()


@pytest.fixture(scope="module")
def quick_check(tmp_path_factory):
    settings = Settings()
    return power.run_check(
        power.PowerConfig.from_mapping(load_study_config(), quick=True),
        master_seed=settings.olab_seed,
        reports_dir=tmp_path_factory.mktemp("reports"),
        quick=True,
    )


@pytest.mark.slow
def test_power_check_quick(quick_check):
    rows, criteria = quick_check
    outside = [r for r in rows if r["test"] != "dsr" and not r["within_tolerance"]]
    assert outside == []
    # Directional at 100 replications: PBO is flat near 0.5 for weak signals, so the rank
    # correlation is noise here. AC-W-02 judges it on the full run.
    medians = criteria["pbo_medians"]
    assert medians[-1] < medians[0] - 0.25


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="AC-W-02 is open: see docs/validation.md")
def test_dsr_matches_its_reference(quick_check):
    rows, _ = quick_check
    assert all(r["within_tolerance"] for r in rows if r["test"] == "dsr")


def test_power_command_reports_cells_outside_tolerance(tmp_path, monkeypatch):
    monkeypatch.setenv("OLAB_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("OLAB_REPORTS_DIR", str(tmp_path / "reports"))
    row = {
        "test": "dsr",
        "true_sharpe_annual": 1.0,
        "years": 5,
        "n_trials": 10,
        "empirical": 0.04,
        "reference": 0.17,
        "within_tolerance": False,
    }
    monkeypatch.setattr(
        power, "run_check", lambda *a, **k: ([row], {"all_within_tolerance": False})
    )
    result = CliRunner().invoke(app, ["sim", "power", "--check", "--quick"])
    assert result.exit_code == 0, result.output
    assert "outside: dsr SR 1.0 5y N=10: 0.040 vs 0.170" in result.stdout


def test_power_command_runs_the_grid(tmp_path, monkeypatch):
    monkeypatch.setenv("OLAB_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("OLAB_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr(power, "run_grid", lambda *a, **k: ([], {"curves_monotone": True}))
    result = CliRunner().invoke(app, ["sim", "power"])
    assert result.exit_code == 0, result.output
    assert "'curves_monotone': True" in result.stdout
