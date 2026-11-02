from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import overfitlab as ol
from overfitlab.snooping import stepdown_statistics
from overfitlab_study.cli import app
from overfitlab_study.validation import studentised_xcheck as sx
from overfitlab_study.validation.arch_xcheck import Case, case_data

CASE = Case(1, 5, 250, False)


def bsds_values(**changes):
    keys = ("c", "u", "l", "rc_u", "same_c", "same_u", "same_l", "same_rc_u")
    return {**dict.fromkeys(keys, 0.5), **changes}


def ours_values(**changes):
    row = {"rc_ours": 0.5, "rc_same_ours": 0.5, "stepm_ours": [], "all_means_negative": False}
    for kind in sx.KINDS:
        row[f"spa_{kind}_ours"] = 0.5
        row[f"spa_{kind}_same_ours"] = 0.5
    return {**row, **changes}


def rows_with(*items):
    return [sx.combine(CASE, 100, ours, bsds, stepwise) for ours, bsds, stepwise in items]


@pytest.mark.skipif(not Path("D:/").exists(), reason="needs a drive letter")
def test_windows_path_to_wsl():
    assert sx.to_wsl(Path("D:/repo/data")) == "/mnt/d/repo/data"


def test_linux_path_passes_through(monkeypatch):
    class Plain:
        drive = ""

        def resolve(self):
            return self

        def as_posix(self):
            return "/home/me/data"

    assert sx.to_wsl(Plain()) == "/home/me/data"


def test_command_uses_wsl_on_windows(monkeypatch):
    monkeypatch.setattr(sx.sys, "platform", "win32")
    assert sx.command(["octave-cli"])[:3] == ["wsl.exe", "-d", "Ubuntu"]
    monkeypatch.setattr(sx.sys, "platform", "linux")
    assert sx.command(["octave-cli"]) == ["octave-cli"]


def test_export_round_trip(tmp_path):
    sx.export_case(CASE, 40, tmp_path)
    d = case_data(CASE)
    stored = np.fromfile(tmp_path / "d.bin", dtype="<f8").reshape(d.shape, order="F")
    assert np.array_equal(stored, d)
    indices = np.fromfile(tmp_path / "indices.bin", dtype="<i4")
    assert indices.size == 40 * 250
    assert (indices.min(), indices.max()) == (1, 250)
    assert (tmp_path / "meta.csv").read_text("utf-8").split(",")[:3] == ["250", "5", "40"]


def test_exported_statistics_are_the_ones_the_stepdown_uses(tmp_path):
    sx.export_case(CASE, 40, tmp_path)
    plan = np.fromfile(tmp_path / "indices.bin", dtype="<i4").reshape(40, 250).astype(np.int64) - 1
    tstats, simulated = stepdown_statistics(case_data(CASE), plan, 5.0)
    assert np.array_equal(np.fromfile(tmp_path / "tstats.bin", dtype="<f8"), tstats)
    stored = np.fromfile(tmp_path / "zstats.bin", dtype="<f8").reshape(40, 5)
    assert np.array_equal(stored, simulated)


def test_our_side_reads_the_exported_resamples(tmp_path):
    sx.export_case(CASE, 40, tmp_path)
    row = sx.our_side(CASE, 40, tmp_path)
    plan = np.fromfile(tmp_path / "indices.bin", dtype="<i4").reshape(40, 250).astype(np.int64) - 1
    expected = ol.spa_test(case_data(CASE), indices=plan, block_length=5)
    assert row["spa_consistent_same_ours"] == expected.pvalue_consistent


def test_read_external_outputs(tmp_path):
    (tmp_path / "bsds.csv").write_text("c,u\n0.25,0.5\n", encoding="utf-8")
    (tmp_path / "stepwise.txt").write_text("0 3\n", encoding="utf-8")
    assert sx.read_bsds(tmp_path) == {"c": 0.25, "u": 0.5}
    assert sx.read_stepwise(tmp_path) == [0, 3]
    (tmp_path / "stepwise.txt").write_text("\n", encoding="utf-8")
    assert sx.read_stepwise(tmp_path) == []


def test_combine_maps_each_p_value():
    row = sx.combine(CASE, 100, ours_values(), bsds_values(l=0.1, c=0.2, u=0.3, same_l=0.4), [])
    assert (row["spa_lower_bsds"], row["spa_consistent_bsds"], row["spa_upper_bsds"]) == (
        0.1,
        0.2,
        0.3,
    )
    assert row["spa_lower_same_bsds"] == 0.4
    assert row["stepm_match"]


def test_criteria_pass_when_everything_agrees():
    rows = rows_with(*[(ours_values(), bsds_values(), []) for _ in range(20)])
    criteria = sx.check_criteria(rows)
    assert criteria["pvalues_within_tolerance"]
    assert criteria["identical_indices_agree"]
    assert criteria["stepm_matches"] == 20
    assert criteria["floor_cases"] == []


def test_floor_case_fails_the_limit_and_is_reported_apart():
    floor = (ours_values(all_means_negative=True), bsds_values(c=1.0, same_c=1.0), [])
    rows = rows_with(floor, (ours_values(), bsds_values(), []))
    criteria = sx.check_criteria(rows)
    assert not criteria["pvalues_within_tolerance"]
    assert not criteria["identical_indices_agree"]
    assert criteria["max_pvalue_diff_without_floor_cases"] == 0.0
    assert criteria["max_identical_diff_without_floor_cases"] == 0.0
    assert len(criteria["floor_cases"]) == 1


def test_report_is_written(tmp_path):
    target = tmp_path / "x" / "studentised.csv"
    sx.write_report(rows_with((ours_values(), bsds_values(), [])), target)
    assert target.read_text("utf-8").startswith("case,n_strategies")


def test_run_case_joins_the_three_sides(tmp_path, monkeypatch):
    def fake_octave(folder, _):
        keys = ",".join(bsds_values())
        (folder / "bsds.csv").write_text(f"{keys}\n" + ",".join(["0.5"] * 8) + "\n", "utf-8")

    def fake_r(folder):
        (folder / "stepwise.txt").write_text("\n", encoding="utf-8")

    monkeypatch.setattr(sx, "run_octave", fake_octave)
    monkeypatch.setattr(sx, "run_r", fake_r)
    row = sx.run_case(CASE, 40, tmp_path, Path("."))
    assert row["case"] == 1
    assert row["stepm_stepwise"] == ""


def test_external_runs_use_the_scripts(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sx.subprocess, "run", lambda args, **_: calls.append(args))
    sx.run_octave(tmp_path, tmp_path)
    sx.run_r(tmp_path)
    assert any("run_bsds.m" in part for part in calls[0])
    assert any("run_stepwise.R" in part for part in calls[1])


def test_toolbox_is_pinned(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sx.subprocess, "run", lambda args, **_: calls.append(args))
    assert sx.ensure_toolbox(tmp_path / "mfe") == tmp_path / "mfe" / "bootstrap"
    assert calls[0][:3] == ["git", "clone", "-q"]
    assert calls[1][-1] == sx.MFE_COMMIT


def test_command_writes_the_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OLAB_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("OLAB_REPORTS_DIR", str(tmp_path / "reports"))
    rows = rows_with((ours_values(), bsds_values(), []))
    monkeypatch.setattr(sx, "run_all", lambda *a, **k: rows)
    result = CliRunner().invoke(app, ["validate", "studentised", "--n-boot", "100"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "reports/validation/studentised_xcheck.csv").exists()
