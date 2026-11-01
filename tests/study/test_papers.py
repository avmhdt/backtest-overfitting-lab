import csv
from pathlib import Path

from typer.testing import CliRunner

from overfitlab_study.cli import app
from overfitlab_study.validation.papers import (
    CaseResult,
    evaluate_case,
    load_cases,
    write_report,
)

GOLDEN = Path(__file__).parents[1] / "golden" / "paper_examples.yaml"


def result(printed, computed, tolerance=0.001):
    return CaseResult("x", "G", "s", "sourced", printed, computed, tolerance, "")


def test_passes_within_tolerance():
    assert result(1.0, 1.0005).passed


def test_fails_outside_tolerance():
    assert not result(1.0, 1.01).passed


def test_pending_case_never_passes():
    pending = next(c for c in load_cases(GOLDEN) if c["status"] == "pending-source")
    evaluated = evaluate_case(pending)
    assert evaluated.computed is None
    assert not evaluated.passed


def test_report_columns_and_status(tmp_path):
    path = tmp_path / "out" / "paper_examples.csv"
    pending = CaseResult("y", "G", "s", "pending-source", None, None, 0.0, "n")
    write_report([result(1.0, 1.0), pending], path)
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == [
        "id",
        "source",
        "printed",
        "computed",
        "abs_diff",
        "tolerance",
        "pass",
        "notes",
    ]
    assert rows[0]["pass"] == "true"
    assert rows[1]["pass"] == "pending-source"


def test_validate_papers_command(tmp_path, monkeypatch):
    golden = tmp_path / "tests" / "golden"
    golden.mkdir(parents=True)
    (golden / "paper_examples.yaml").write_text(GOLDEN.read_text("utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLAB_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("OLAB_REPORTS_DIR", str(tmp_path / "reports"))
    output = CliRunner().invoke(app, ["validate", "papers"])
    assert output.exit_code == 0, output.output
    assert "known miss G-03c" in output.stdout
    assert (tmp_path / "reports" / "validation" / "paper_examples.csv").exists()
    assert (tmp_path / "results" / "experiment_log.csv").exists()


def test_validate_papers_fails_on_an_unexpected_miss(tmp_path, monkeypatch):
    golden = tmp_path / "tests" / "golden"
    golden.mkdir(parents=True)
    text = GOLDEN.read_text("utf-8").replace("printed: 0.9004", "printed: 0.5")
    (golden / "paper_examples.yaml").write_text(text, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLAB_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("OLAB_REPORTS_DIR", str(tmp_path / "reports"))
    assert CliRunner().invoke(app, ["validate", "papers"]).exit_code == 1
