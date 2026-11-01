import csv
import json

import pytest

from overfitlab_study.explog import (
    EXPERIMENT_LOG_COLUMNS,
    canonical_json,
    config_hash,
    git_sha,
    study_run,
)
from overfitlab_study.settings import Settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLAB_RESULTS_DIR", str(tmp_path / "results"))
    return Settings()


def read_log(settings):
    path = settings.olab_results_dir / "experiment_log.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_row_written_with_metrics(settings):
    with study_run("sim noise", settings, params={"seeds": 5}) as run:
        run.metrics["best"] = 1.5
    [row] = read_log(settings)
    assert tuple(row) == EXPERIMENT_LOG_COLUMNS
    assert row["command"] == "sim noise"
    assert json.loads(row["params_json"]) == {"seeds": 5}
    assert json.loads(row["metrics_json"]) == {"best": 1.5, "status": "ok"}
    assert float(row["duration_s"]) >= 0


def test_failed_run_is_still_logged(settings):
    with pytest.raises(RuntimeError), study_run("bad", settings):
        raise RuntimeError("boom")
    [row] = read_log(settings)
    assert json.loads(row["metrics_json"])["status"] == "failed"


def test_log_is_append_only(settings):
    for _ in range(3):
        with study_run("x", settings):
            pass
    rows = read_log(settings)
    assert len(rows) == 3
    assert len({row["run_id"] for row in rows}) == 3


def test_log_uses_lf(settings):
    with study_run("x", settings):
        pass
    assert b"\r" not in (settings.olab_results_dir / "experiment_log.csv").read_bytes()


def test_json_lines_log(settings):
    with study_run("x", settings) as run:
        pass
    path = settings.olab_results_dir / "logs" / f"{run.run_id}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["run_id"] == run.run_id
    assert records[0]["command"] == "x"
    assert records[0]["seed"] == settings.olab_seed


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_canonical_json_is_compact():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_git_sha_outside_a_repo(tmp_path):
    assert git_sha(tmp_path) == "unknown"
