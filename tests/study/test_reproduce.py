import hashlib

import pytest
from typer.testing import CliRunner

from overfitlab_study.cli import app


def table_hashes(reports):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((reports / "tables").glob("*.csv"))
    }


@pytest.mark.slow
def test_quick_run_is_byte_identical(tmp_path, monkeypatch):
    monkeypatch.setenv("OLAB_DATA_DIR", str(tmp_path / "no-data"))
    hashes = []
    for name in ("first", "second"):
        monkeypatch.setenv("OLAB_RESULTS_DIR", str(tmp_path / name / "results"))
        monkeypatch.setenv("OLAB_REPORTS_DIR", str(tmp_path / name / "reports"))
        result = CliRunner().invoke(app, ["all", "--quick"])
        assert result.exit_code == 0, result.output
        assert "skipped: case studies" in result.stdout
        hashes.append(table_hashes(tmp_path / name / "reports"))
    assert hashes[0] == hashes[1]
    assert {"noise_experiment.csv", "leakage.csv", "power_check.csv"} <= set(hashes[0])
    log = (tmp_path / "second" / "results" / "experiment_log.csv").read_text("utf-8")
    assert "sim noise" in log
