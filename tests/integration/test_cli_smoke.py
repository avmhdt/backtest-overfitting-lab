import json
from importlib.metadata import entry_points

import numpy as np
import pytest
from typer.testing import CliRunner

import overfitlab
from overfitlab.cli import _clean, app
from overfitlab_study.cli import app as study_app

runner = CliRunner()


def invoke_json(*args):
    result = runner.invoke(app, [*args, "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == overfitlab.__version__


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["psr", "--help"],
        ["dsr", "--help"],
        ["mintrl", "--help"],
        ["minbtl", "--help"],
        ["haircut", "--help"],
        ["pbo", "--help"],
        ["snoop", "--help"],
        ["trials", "--help"],
        ["trials", "validate", "--help"],
        ["trials", "report", "--help"],
        ["trials", "add-manual", "--help"],
    ],
)
def test_library_help(args):
    assert runner.invoke(app, args).exit_code == 0


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["data", "--help"],
        ["data", "smoke", "--help"],
        ["validate", "papers", "--help"],
        ["validate", "arch", "--help"],
        ["sim", "leakage", "--help"],
        ["sim", "noise", "--help"],
        ["bench", "--help"],
    ],
)
def test_study_help(args):
    assert runner.invoke(study_app, args).exit_code == 0


def test_dsr_paper_case():
    out = invoke_json(
        "dsr",
        "--sr-annual=2.5",
        "--periods-per-year=250",
        "--n-obs=1250",
        "--skew=-3",
        "--kurtosis=10",
        "--n-trials=100",
        "--var-trials-annual=0.5",
    )
    assert out["sr0"] == pytest.approx(0.1132, abs=5e-4)
    assert out["dsr"] == pytest.approx(0.9004, abs=5e-4)
    assert "sr0_annual" in out
    assert "z" in out


def test_dsr_per_period_inputs():
    out = invoke_json("dsr", "--sr=0.1", "--n-obs=500", "--n-trials=10", "--var-trials=0.001")
    assert "sr0_annual" not in out


def test_psr_per_period_and_annual_agree():
    per_period = invoke_json("psr", "--sr=0.458", "--n-obs=24")
    annual = invoke_json(
        "psr", f"--sr-annual={0.458 * 12**0.5}", "--periods-per-year=12", "--n-obs=24"
    )
    assert per_period["psr"] == pytest.approx(annual["psr"])
    assert "sr_std" in per_period


def test_psr_benchmark_uses_the_units_of_the_sr():
    out = invoke_json(
        "psr", "--sr-annual=1", "--periods-per-year=12", "--n-obs=24", "--benchmark=1"
    )
    assert out["psr"] == pytest.approx(0.5)


def test_mintrl():
    out = invoke_json("mintrl", "--sr-annual=2", "--periods-per-year=12", "--benchmark-annual=1")
    assert out["years"] == pytest.approx(3.24, abs=0.005)


def test_minbtl():
    assert invoke_json("minbtl", "--n-trials=45")["years"] == pytest.approx(5.0, abs=0.5)


def test_haircut():
    out = invoke_json(
        "haircut",
        "--sr-annual=1",
        "--n-obs=120",
        "--n-tests=100",
        "--method=bonferroni",
        "--autocorr=0.1",
        "--rho-tests=0.4",
    )
    assert out["haircut_sr_annual"] == pytest.approx(0.232, abs=5e-4)


def test_table_output_without_json():
    result = runner.invoke(app, ["minbtl", "--n-trials=45"])
    assert result.exit_code == 0
    assert result.stdout.startswith("years:")


@pytest.mark.parametrize(
    "args",
    [
        ["psr", "--n-obs=24"],
        ["psr", "--n-obs=24", "--sr=0.1", "--sr-annual=0.1"],
        ["psr", "--n-obs=24", "--sr-annual=1"],
        ["psr", "--n-obs=1", "--sr=0.1"],
        ["dsr", "--n-obs=24", "--sr=0.1", "--n-trials=5"],
        ["dsr", "--n-obs=24", "--sr=0.1", "--n-trials=5", "--var-trials-annual=0.5"],
        ["mintrl", "--sr-annual=-1", "--periods-per-year=12"],
        ["minbtl", "--n-trials=0"],
        ["haircut", "--sr-annual=1", "--n-obs=1", "--n-tests=5"],
    ],
)
def test_bad_arguments_exit_non_zero(args):
    assert runner.invoke(app, args).exit_code == 1


@pytest.fixture
def wide(tmp_path):
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 0.01, (320, 4))
    values[:, 1] += 0.01
    days = np.datetime64("2023-01-02", "D") + np.arange(320)
    lines = ["date,a,b,c,base"]
    for day, row in zip(days, values, strict=True):
        lines.append(",".join([str(day), *(f"{v:.8f}" for v in row)]))
    path = tmp_path / "wide.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_pbo_command(wide):
    out = invoke_json("pbo", wide, "--partitions=8")
    assert out["n_combinations"] == 70
    assert 0.0 <= out["pbo"] <= 1.0


@pytest.mark.parametrize("test", ["rc", "spa", "stepm"])
def test_snoop_command_with_a_zero_benchmark(wide, test):
    out = invoke_json("snoop", wide, "--zero-benchmark", f"--test={test}", "--n-boot=100")
    assert out


def test_snoop_with_a_benchmark_column_drops_that_column(wide):
    out = invoke_json(
        "snoop", wide, "--benchmark-col=base", "--test=stepm", "--n-boot=200", "--block-length=2"
    )
    assert "base" not in out["rejected"]
    assert "b" in out["rejected"]


@pytest.mark.parametrize(
    "args",
    [
        ["--zero-benchmark", "--benchmark-col=a"],
        [],
        ["--benchmark-col=nope"],
        ["--zero-benchmark", "--test=other"],
    ],
)
def test_snoop_rejects_bad_arguments(wide, args):
    assert runner.invoke(app, ["snoop", wide, *args]).exit_code == 1


def test_pbo_rejects_bad_partitions(wide):
    assert runner.invoke(app, ["pbo", wide, "--partitions=3"]).exit_code == 1


def test_trials_validate_clean(materialise):
    log_path = materialise("trial_log_365")
    out = invoke_json("trials", "validate", str(log_path))
    assert out["n_errors"] == 0


def test_trials_validate_exit_code(materialise):
    log_path = str(materialise("trial_log_bad/E_FAMILY"))
    result = runner.invoke(app, ["trials", "validate", log_path])
    assert result.exit_code == 1
    assert "E_FAMILY" in result.stdout


def test_trials_validate_no_strict_reports_and_exits_zero(materialise):
    log_path = str(materialise("trial_log_bad/E_FAMILY"))
    result = runner.invoke(app, ["trials", "validate", log_path, "--no-strict"])
    assert result.exit_code == 0
    assert "E_FAMILY" in result.stdout


def test_trials_report(materialise, tmp_path):
    log_path = str(materialise("trial_log_365"))
    target = tmp_path / "out" / "report.json"
    out = invoke_json(
        "trials", "report", log_path, "--family", "crypto_daily", "--out", str(target)
    )
    assert out["periods_per_year"] == 365
    assert json.loads(target.read_text("utf-8"))["n_trials"] == 3


def test_trials_report_error_exit(materialise):
    log_path = str(materialise("trial_log_bad/E_FAMILY"))
    assert runner.invoke(app, ["trials", "report", log_path]).exit_code == 1


def test_trials_add_manual(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log_path = tmp_path / "trial_log.csv"
    out = invoke_json(
        "trials",
        "add-manual",
        str(log_path),
        "--family=demo",
        '--params-json={"idea":1}',
        "--note=by hand",
        "--sharpe-annual=1.2",
    )
    assert out["run_id"] in log_path.read_text("utf-8")


def test_trials_add_manual_rejects_bad_json(tmp_path):
    args = ["trials", "add-manual", str(tmp_path / "x.csv"), "--family=demo"]
    assert runner.invoke(app, [*args, "--params-json=oops", "--note=x"]).exit_code == 1


def test_nan_becomes_null_in_json():
    assert _clean({"a": float("nan"), "b": [float("inf"), 1.0]}) == {"a": None, "b": [None, 1.0]}
    assert np.isfinite(_clean(1.0))


@pytest.mark.slow
def test_study_entry_point_resolves():
    names = {ep.name for ep in entry_points(group="console_scripts")}
    assert {"overfitlab", "olab-study"} <= names
