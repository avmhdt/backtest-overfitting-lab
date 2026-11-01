import csv
import hashlib
import importlib.util
import json
import math
import re
import subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import overfitlab as ol
from overfitlab import TrialLogError, TrialLogger, triallog

REPO = Path(__file__).parents[2]
FIXTURES = REPO / "tests" / "fixtures"
BAD = sorted(path.name for path in (FIXTURES / "trial_log_bad").iterdir())


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def codes(path, severity="error"):
    return sorted({i.code for i in ol.validate_trial_log(path) if i.severity == severity})


def edit_cell(path: Path, column: str, value: str, row: int = 0) -> None:
    with path.open(encoding="utf-8", newline="") as fh:
        table = list(csv.reader(fh))
    table[row + 1][table[0].index(column)] = value
    with path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(table)


@pytest.fixture
def logger(tmp_path):
    return TrialLogger(tmp_path / "trial_log.csv", tmp_path / "runs", repo_dir=tmp_path)


def daily(count=100, start="2024-01-01"):
    first = np.datetime64(start, "D")
    return first + np.arange(count)


def test_valid_fixture_has_no_errors(materialise):
    assert codes(materialise("trial_log_example")) == []


def test_valid_fixture_warnings(materialise):
    assert codes(materialise("trial_log_example"), "warning") == ["W_FAILED", "W_SHORT"]


@pytest.mark.parametrize("code", BAD)
def test_bad_fixture_gives_exactly_its_code(materialise, code):
    assert codes(materialise(f"trial_log_bad/{code}")) == [code]


def test_every_error_code_has_a_fixture():
    schema = json.loads((REPO / "src/overfitlab/trial_log_schema.json").read_text("utf-8"))
    assert sorted(schema["errors"]) == BAD


def test_worked_example(materialise):
    hand = load_module(FIXTURES / "trial_log_hand_calc.py", "hand_calc").main()
    report = ol.evaluate_trial_log(materialise("trial_log_example"), families=["ma_demo"])
    assert report.n_trials == hand["N"]
    assert report.n_valid == hand["M"]
    for actual, key in (
        (report.variance, "V"),
        (report.sr0, "SR0"),
        (report.sr, "SR"),
        (report.skew, "skew"),
        (report.kurtosis, "kurtosis"),
        (report.z, "z"),
        (report.dsr, "DSR"),
    ):
        assert actual == pytest.approx(hand[key], abs=1e-9), key


def test_worked_example_in_the_docs_matches_the_hand_calculation():
    hand = load_module(FIXTURES / "trial_log_hand_calc.py", "hand_calc").main()
    text = (REPO / "docs/trial-log.md").read_text("utf-8")
    for key in ("V", "SR0", "SR", "z", "DSR"):
        assert f"{hand[key]:.12f}" in text, key


def test_365_fixture_uses_declared_periods(materialise):
    log_path = materialise("trial_log_365")
    assert codes(log_path) == []
    report = ol.evaluate_trial_log(log_path)
    assert report.periods_per_year == 365
    assert report.variance_annual == pytest.approx(report.variance * 365)
    assert report.sr_annual == pytest.approx(report.sr * math.sqrt(365))


def test_families_at_different_frequencies_may_share_a_log(logger, tmp_path):
    days = daily()
    r = np.random.default_rng(0).normal(0, 0.01, days.size)
    logger.log(strategy_family="crypto", params={}, dates=days, returns=r, periods_per_year=365)
    logger.log(strategy_family="equity", params={}, dates=days, returns=r, periods_per_year=252)
    log_path = tmp_path / "trial_log.csv"
    assert codes(log_path) == []
    assert ol.evaluate_trial_log(log_path, families=["crypto"]).periods_per_year == 365
    with pytest.raises(TrialLogError) as caught:
        ol.evaluate_trial_log(log_path)
    assert caught.value.issues[0].code == "E_FREQ_MIX"


def test_declared_365_next_to_inferred_252_is_a_mix(logger, tmp_path):
    days = daily()
    r = np.random.default_rng(0).normal(0, 0.01, days.size)
    logger.log(strategy_family="demo", params={"n": 1}, dates=days, returns=r, periods_per_year=365)
    logger.log(strategy_family="demo", params={"n": 2}, dates=days, returns=r, periods_per_year=252)
    edit_cell(tmp_path / "trial_log.csv", "x_periods_per_year", "", row=1)
    assert codes(tmp_path / "trial_log.csv") == ["E_FREQ_MIX"]


@pytest.mark.parametrize(("step", "expected"), [(1, 252), (7, 52), (30, 12)])
def test_frequency_inference(logger, tmp_path, step, expected):
    days = np.datetime64("2020-01-01", "D") + np.arange(100) * step
    r = np.random.default_rng(0).normal(0, 0.01, 100)
    logger.log(strategy_family="demo", params={}, dates=days, returns=r, periods_per_year=expected)
    edit_cell(tmp_path / "trial_log.csv", "x_periods_per_year", "")
    assert ol.read_trial_log(tmp_path / "trial_log.csv").periods_per_year() == expected


def test_unknown_spacing_is_a_frequency_error(logger, tmp_path):
    days = np.datetime64("2020-01-01", "D") + np.arange(100) * 15
    r = np.random.default_rng(0).normal(0, 0.01, 100)
    logger.log(strategy_family="demo", params={}, dates=days, returns=r)
    edit_cell(tmp_path / "trial_log.csv", "x_periods_per_year", "")
    assert codes(tmp_path / "trial_log.csv") == ["E_FREQ_MIX"]


def test_declared_periods_allow_any_spacing(logger, tmp_path):
    days = np.datetime64("2020-01-01", "D") + np.arange(100) * 91
    r = np.random.default_rng(0).normal(0, 0.01, 100)
    logger.log(strategy_family="demo", params={}, dates=days, returns=r, periods_per_year=4)
    assert codes(tmp_path / "trial_log.csv") == []


def test_sharpe_mismatch_warns(logger, tmp_path):
    days = daily()
    logger.log(
        strategy_family="demo",
        params={},
        dates=days,
        returns=np.random.default_rng(0).normal(0, 0.01, days.size),
    )
    edit_cell(tmp_path / "trial_log.csv", "sharpe_annual", "9.5")
    assert "W_SHARPE_MISMATCH" in codes(tmp_path / "trial_log.csv", "warning")


def test_dirty_row_warns(materialise):
    log_path = materialise("trial_log_365")
    sha = "0123456789abcdef0123456789abcdef01234567-dirty"
    edit_cell(log_path, "git_sha", sha)
    assert codes(log_path, "warning") == ["W_DIRTY"]


def test_timestamp_with_impossible_date(materialise):
    log_path = materialise("trial_log_365")
    edit_cell(log_path, "timestamp", "2026-13-40T12:00:00Z")
    assert codes(log_path) == ["E_TIMESTAMP"]


def test_uppercase_run_id_rejected(materialise):
    log_path = materialise("trial_log_365")
    with log_path.open(encoding="utf-8") as fh:
        first = fh.read().splitlines()[1].split(",")[0]
    edit_cell(log_path, "run_id", first.upper())
    assert "E_RUN_ID" in codes(log_path)


def test_bom_is_rejected(tmp_path, materialise):
    log_path = materialise("trial_log_365")
    log_path.write_bytes(b"\xef\xbb\xbf" + log_path.read_bytes())
    assert codes(log_path) == ["E_COLUMNS"]


def test_binary_file_is_rejected(tmp_path):
    path = tmp_path / "log.csv"
    path.write_bytes(b"\xff\xfe\x00\x01")
    assert codes(path) == ["E_COLUMNS"]


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text("", encoding="utf-8")
    assert codes(path) == ["E_COLUMNS"]


def test_unknown_column_is_rejected(tmp_path, materialise):
    log_path = materialise("trial_log_365")
    with log_path.open(encoding="utf-8", newline="") as fh:
        table = list(csv.reader(fh))
    table[0].append("note")
    for row in table[1:]:
        row.append("")
    with log_path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(table)
    assert codes(log_path) == ["E_COLUMNS"]


def test_short_row_is_rejected(materialise):
    log_path = materialise("trial_log_365")
    with log_path.open("a", encoding="utf-8", newline="") as fh:
        fh.write("only,two\n")
    assert codes(log_path) == ["E_COLUMNS"]


def test_unknown_x_columns_are_ignored(logger, tmp_path):
    days = daily()
    logger.log(
        strategy_family="demo",
        params={},
        dates=days,
        returns=np.random.default_rng(0).normal(0, 0.01, days.size),
        extra={"x_dataset": "synthetic"},
    )
    assert codes(tmp_path / "trial_log.csv") == []


def test_fresh_clone_is_summary_only(materialise):
    log_path = materialise("trial_log_365")
    for path in (log_path.parent / "runs").rglob("*.parquet"):
        path.unlink()
    assert codes(log_path) == []
    assert codes(log_path, "warning") == ["W_SUMMARY_ONLY"]
    report = ol.evaluate_trial_log(log_path)
    assert report.variance_source == "summary"
    assert report.moments_assumed
    assert (report.skew, report.kurtosis) == (0.0, 3.0)
    assert report.periods_per_year == 365


def test_summary_mode_infers_frequency_without_declared_periods(materialise):
    log_path = materialise("trial_log_365")
    for path in (log_path.parent / "runs").rglob("*.parquet"):
        path.unlink()
    for row in range(3):
        edit_cell(log_path, "x_periods_per_year", "", row=row)
    assert ol.read_trial_log(log_path).periods_per_year() == 252


def test_summary_only_unreadable_frequency(materialise):
    log_path = materialise("trial_log_365")
    for path in (log_path.parent / "runs").rglob("*.parquet"):
        path.unlink()
    edit_cell(log_path, "x_periods_per_year", "", row=0)
    edit_cell(log_path, "end", "2024-01-01", row=0)
    edit_cell(log_path, "n_obs", "1", row=0)
    assert "E_FREQ_MIX" in codes(log_path)


def test_timestamp_date_column_is_accepted(logger, tmp_path):
    days = daily()
    r = np.random.default_rng(0).normal(0, 0.01, days.size)
    run_id = logger.log(strategy_family="demo", params={}, dates=days, returns=r)
    stamps = pa.array(days.astype("datetime64[ms]"), pa.timestamp("ms"))
    pq.write_table(
        pa.table({"date": stamps, "ret": r, "ret_gross": r}),
        tmp_path / "runs" / run_id / "returns.parquet",
    )
    assert codes(tmp_path / "trial_log.csv") == []


def test_timestamp_with_time_is_rejected(logger, tmp_path):
    days = daily()
    r = np.random.default_rng(0).normal(0, 0.01, days.size)
    run_id = logger.log(strategy_family="demo", params={}, dates=days, returns=r)
    stamps = (days.astype("datetime64[ms]") + np.timedelta64(5, "h")).astype("datetime64[ms]")
    pq.write_table(
        pa.table({"date": pa.array(stamps, pa.timestamp("ms")), "ret": r}),
        tmp_path / "runs" / run_id / "returns.parquet",
    )
    assert codes(tmp_path / "trial_log.csv") == ["E_RETURNS_SCHEMA"]


def test_float32_returns_are_rejected(logger, tmp_path):
    days = daily()
    r = np.random.default_rng(0).normal(0, 0.01, days.size)
    run_id = logger.log(strategy_family="demo", params={}, dates=days, returns=r)
    table = pa.table({"date": pa.array(days, pa.date32()), "ret": pa.array(r, pa.float32())})
    pq.write_table(table, tmp_path / "runs" / run_id / "returns.parquet")
    assert codes(tmp_path / "trial_log.csv") == ["E_RETURNS_SCHEMA"]


def test_nan_returns_are_rejected(logger, tmp_path):
    days = daily()
    r = np.random.default_rng(0).normal(0, 0.01, days.size)
    run_id = logger.log(strategy_family="demo", params={}, dates=days, returns=r)
    r[3] = np.nan
    table = pa.table({"date": pa.array(days, pa.date32()), "ret": pa.array(r, from_pandas=False)})
    pq.write_table(table, tmp_path / "runs" / run_id / "returns.parquet")
    assert codes(tmp_path / "trial_log.csv") == ["E_RETURNS_VALUES"]


def test_logger_writes_the_dialect(logger, tmp_path):
    days = daily()
    logger.log(
        strategy_family="demo",
        params={"b": [1, 2], "a": "x,y"},
        dates=days,
        returns=np.random.default_rng(0).normal(0, 0.01, days.size),
    )
    raw = (tmp_path / "trial_log.csv").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    header = raw.split(b"\n")[0].decode().split(",")
    assert tuple(header[:10]) == ol.TRIAL_LOG_COLUMNS
    assert header[10:] == ["x_periods_per_year", "x_note"]
    row = next(csv.DictReader(raw.decode().splitlines()))
    assert row["params_json"] == '{"a":"x,y","b":[1,2]}'


def test_logger_records_six_significant_digits(logger, tmp_path):
    days = daily()
    r = np.random.default_rng(0).normal(0, 0.01, days.size)
    logger.log(strategy_family="demo", params={}, dates=days, returns=r)
    with (tmp_path / "trial_log.csv").open(encoding="utf-8", newline="") as fh:
        row = next(csv.DictReader(fh))
    digits = re.sub(r"[-.]|e.*$", "", row["sharpe_annual"]).lstrip("0")
    assert len(digits) >= 6


def test_logger_keeps_existing_header(logger, tmp_path):
    days = daily()
    r = np.random.default_rng(0).normal(0, 0.01, days.size)
    logger.log(strategy_family="demo", params={}, dates=days, returns=r, extra={"x_a": "1"})
    logger.log(strategy_family="demo", params={}, dates=days, returns=r)
    with pytest.raises(ValueError, match="append-only"):
        logger.log(strategy_family="demo", params={}, dates=days, returns=r, extra={"x_b": "1"})
    assert len(ol.read_trial_log(tmp_path / "trial_log.csv").rows) == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"strategy_family": "Bad"}, "strategy_family"),
        ({"periods_per_year": 0}, "periods_per_year"),
        ({"returns": [0.1, 0.2]}, "length"),
        ({"dates": ["2024-01-01"], "returns": [0.1]}, "two observations"),
        ({"dates": ["2024-01-02", "2024-01-01"], "returns": [0.1, 0.2]}, "increasing"),
        ({"dates": ["2024-01-01", "2024-01-01"], "returns": [0.1, 0.2]}, "increasing"),
        ({"extra": {"note": "x"}}, "x_"),
    ],
)
def test_logger_rejects_contract_breaks(logger, kwargs, message):
    arguments = {
        "strategy_family": "demo",
        "params": {},
        "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "returns": [0.1, 0.2, 0.3],
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        logger.log(**arguments)


def test_logger_rejects_nan_params(logger):
    with pytest.raises(ValueError, match="JSON"):
        logger.log(
            strategy_family="demo",
            params={"x": float("nan")},
            dates=["2024-01-01", "2024-01-02"],
            returns=[0.1, 0.2],
        )


def test_logger_needs_runs_next_to_the_log(tmp_path):
    with pytest.raises(ValueError, match="runs"):
        TrialLogger(tmp_path / "a" / "trial_log.csv", tmp_path / "b" / "runs")


def test_failed_run_counts_in_n_not_in_v(logger, tmp_path):
    days = daily()
    rng = np.random.default_rng(0)
    for n in range(3):
        r = rng.normal(0, 0.01, days.size)
        logger.log(strategy_family="demo", params={"n": n}, dates=days, returns=r)
    logger.log_failed(strategy_family="demo", params={"n": 3}, note="crashed")
    log = ol.read_trial_log(tmp_path / "trial_log.csv")
    assert log.n_trials() == 4
    assert log.trial_sharpes().sr.size == 3
    assert log.filter(exclude_failed=True).n_trials() == 3


def test_log_failed_validates_family(logger):
    with pytest.raises(ValueError, match="strategy_family"):
        logger.log_failed(strategy_family="Bad", params={}, note="x")


def test_manual_trial_counts_in_n_not_in_v(logger, tmp_path):
    days = daily()
    r = np.random.default_rng(0).normal(0, 0.01, days.size)
    logger.log(strategy_family="demo", params={"n": 1}, dates=days, returns=r)
    logger.log(strategy_family="demo", params={"n": 2}, dates=days, returns=r * 0.5 + 0.001)
    ol.add_manual_trial(
        tmp_path / "trial_log.csv",
        strategy_family="demo",
        params={"idea": "by hand"},
        note="looked bad",
        start="2024-01-01",
        end="2024-03-01",
        sharpe_annual=9.0,
        repo_dir=tmp_path,
    )
    log_path = tmp_path / "trial_log.csv"
    assert codes(log_path) == []
    log = ol.read_trial_log(log_path)
    assert log.n_trials() == 3
    assert log.trial_sharpes().sr.size == 2
    assert log.rows[log.selected_index()]["n_obs"] != "0"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"strategy_family": "Bad"}, "strategy_family"), ({"start": "2024-13-01"}, "date")],
)
def test_add_manual_validates(tmp_path, kwargs, message):
    arguments = {"strategy_family": "demo", "params": {}, "note": "x", **kwargs}
    with pytest.raises(ValueError, match=message):
        ol.add_manual_trial(tmp_path / "trial_log.csv", repo_dir=tmp_path, **arguments)


def test_dedup_counts_identical_trials_once(logger, tmp_path):
    days = daily()
    r = np.random.default_rng(0).normal(0, 0.01, days.size)
    for _ in range(3):
        logger.log(strategy_family="demo", params={"n": 1}, dates=days, returns=r)
    log = ol.read_trial_log(tmp_path / "trial_log.csv")
    assert log.n_trials() == 3
    assert log.n_trials(mode="dedup") == 1


def test_unknown_count_mode(materialise):
    with pytest.raises(ValueError, match="mode"):
        ol.read_trial_log(materialise("trial_log_365")).n_trials(mode="other")


def test_variance_fallback_with_one_trial(logger, tmp_path):
    days = daily(101)
    r = np.random.default_rng(0).normal(0.001, 0.01, days.size)
    logger.log(strategy_family="demo", params={}, dates=days, returns=r)
    report = ol.evaluate_trial_log(tmp_path / "trial_log.csv")
    assert report.variance == pytest.approx(0.01)
    assert report.variance_fallback


def test_no_valid_trial_cannot_be_evaluated(logger, tmp_path):
    logger.log_failed(strategy_family="demo", params={}, note="x")
    with pytest.raises(ValueError, match="track record"):
        ol.evaluate_trial_log(tmp_path / "trial_log.csv")


def test_selection_order(logger, tmp_path):
    days = daily()
    rng = np.random.default_rng(0)
    ids = [
        logger.log(
            strategy_family="demo",
            params={"n": n},
            dates=days,
            returns=rng.normal(drift, 0.01, days.size),
            extra={"x_selected": str(n == 0).lower()},
        )
        for n, drift in enumerate((0.0, 0.002))
    ]
    log = ol.read_trial_log(tmp_path / "trial_log.csv")
    assert log.rows[log.selected_index()]["run_id"] == ids[0]
    assert log.rows[log.selected_index(ids[1])]["run_id"] == ids[1]
    edit_cell(tmp_path / "trial_log.csv", "x_selected", "false")
    log = ol.read_trial_log(tmp_path / "trial_log.csv")
    assert log.rows[log.selected_index()]["run_id"] == ids[1]


def test_selecting_unknown_run_fails(materialise):
    with pytest.raises(ValueError, match="not in scope"):
        ol.read_trial_log(materialise("trial_log_365")).selected_index("nope")


def test_selecting_nothing_fails(logger, tmp_path):
    logger.log_failed(strategy_family="demo", params={}, note="x")
    with pytest.raises(ValueError, match="select"):
        ol.read_trial_log(tmp_path / "trial_log.csv").selected_index()


def test_selected_trial_needs_a_series(logger, tmp_path):
    days = daily()
    r = np.random.default_rng(0).normal(0, 0.01, days.size)
    logger.log(strategy_family="demo", params={}, dates=days, returns=r)
    failed = logger.log_failed(strategy_family="demo", params={}, note="x")
    with pytest.raises(ValueError, match="track record"):
        ol.evaluate_trial_log(tmp_path / "trial_log.csv", selected=failed)


def test_returns_matrix_is_an_inner_join(logger, tmp_path):
    rng = np.random.default_rng(0)
    first = logger.log(
        strategy_family="demo",
        params={"n": 1},
        dates=daily(50),
        returns=rng.normal(0, 0.01, 50),
    )
    second = logger.log(
        strategy_family="demo",
        params={"n": 2},
        dates=daily(50, "2024-01-21"),
        returns=rng.normal(0, 0.01, 50),
    )
    dates, matrix, run_ids = ol.read_trial_log(tmp_path / "trial_log.csv").returns_matrix()
    assert run_ids == (first, second)
    assert matrix.shape == (30, 2)
    assert str(dates[0]) == "2024-01-21"


def test_returns_matrix_of_summary_log_is_empty(materialise):
    log_path = materialise("trial_log_365")
    for path in (log_path.parent / "runs").rglob("*.parquet"):
        path.unlink()
    _, matrix, run_ids = ol.read_trial_log(log_path).returns_matrix()
    assert matrix.shape == (0, 0)
    assert run_ids == ()


def test_strict_read_raises_with_issues(materialise):
    with pytest.raises(TrialLogError) as caught:
        ol.read_trial_log(materialise("trial_log_bad/E_FAMILY"))
    assert [issue.code for issue in caught.value.issues] == ["E_FAMILY"]


def test_non_strict_read_keeps_going(materialise):
    assert len(ol.read_trial_log(materialise("trial_log_bad/E_FAMILY"), strict=False).rows) == 1


def test_read_without_validation(materialise):
    log = ol.read_trial_log(materialise("trial_log_bad/E_FAMILY"), validate=False)
    assert len(log.rows) == 1


def test_short_sample_reports_haircut_not_computed(materialise):
    report = ol.evaluate_trial_log(materialise("trial_log_example"), families=["rev_demo"])
    assert report.haircut_sr_annual is None
    assert report.reasons["haircut"].startswith("not computed")


def test_report_json_round_trip(materialise):
    report = ol.evaluate_trial_log(materialise("trial_log_365"))
    assert json.loads(report.to_json())["n_trials"] == report.n_trials


def test_git_sha_ignores_the_log_itself(tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    trials = TrialLogger(
        tmp_path / "results/trial_log.csv", tmp_path / "results/runs", repo_dir=tmp_path
    )
    trials.log_failed(strategy_family="demo", params={"n": 1}, note="first")
    git("add", ".")
    git("commit", "-m", "start")
    trials.log_failed(strategy_family="demo", params={"n": 2}, note="second")
    trials.log_failed(strategy_family="demo", params={"n": 3}, note="third")
    with (tmp_path / "results/trial_log.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[1]["git_sha"].endswith("-dirty") is False
    assert rows[2]["git_sha"].endswith("-dirty") is False
    (tmp_path / "code.py").write_text("x = 2\n", encoding="utf-8")
    trials.log_failed(strategy_family="demo", params={"n": 4}, note="fourth")
    with (tmp_path / "results/trial_log.csv").open(encoding="utf-8", newline="") as fh:
        assert list(csv.DictReader(fh))[3]["git_sha"].endswith("-dirty")


def test_schema_matches_the_code():
    schema = json.loads((REPO / "src/overfitlab/trial_log_schema.json").read_text("utf-8"))
    columns = [column for column in schema["columns"] if column["required"]]
    assert tuple(column["name"] for column in columns) == ol.TRIAL_LOG_COLUMNS
    assert schema["version"] == ol.TRIAL_LOG_VERSION
    by_name = {column["name"]: column for column in columns}
    assert by_name["run_id"]["regex"] == triallog._RUN_ID.pattern
    assert by_name["timestamp"]["regex"] == triallog._TIMESTAMP.pattern
    assert by_name["git_sha"]["regex"] == triallog._GIT_SHA.pattern
    assert by_name["strategy_family"]["regex"] == triallog._FAMILY.pattern


def test_contract_is_frozen():
    # Contract v1.0 is frozen (D-11). If this fails, the change needs the owner's approval
    # and a minor version bump. Only additive x_ columns are allowed.
    schema = json.loads((REPO / "src/overfitlab/trial_log_schema.json").read_text("utf-8"))
    frozen = {key: schema[key] for key in ("version", "csv", "parquet", "errors")}
    frozen["required"] = [c for c in schema["columns"] if c["required"]]
    digest = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    assert digest == "1553b1d5cd264946224b8f336ad71597dd78601aee7fe59fcd71a43e36011f93"


def test_reference_writer_in_the_docs_passes_the_validator(tmp_path):
    text = (REPO / "docs/trial-log.md").read_text("utf-8")
    code = re.search(r"<!-- reference-writer -->\n```python\n(.*?)```", text, re.S)
    assert code is not None
    source = tmp_path / "writer.py"
    source.write_text(code.group(1), encoding="utf-8")
    writer = load_module(source, "reference_writer")
    assert code.group(1).count("\n") <= 110

    log_path = tmp_path / "results" / "trial_log.csv"
    rng = np.random.default_rng(0)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    for n in range(2):
        returns = rng.normal(0.0, 0.01, len(dates)).tolist()
        writer.log_trial(log_path, "crypto_daily", {"n": n}, dates, returns, 365)
    writer.log_failed(log_path, "crypto_daily", {"n": 9}, "aborted")

    assert codes(log_path) == []
    report = ol.evaluate_trial_log(log_path)
    assert report.n_trials == 3
    assert report.periods_per_year == 365
