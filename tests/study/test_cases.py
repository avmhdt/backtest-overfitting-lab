import csv
import datetime as dt

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

import overfitlab as ol
from overfitlab_study.cases import factors, osap, projects, stooq_factory
from overfitlab_study.cases import universe as universe_module
from overfitlab_study.cli import app
from overfitlab_study.data import french, tables
from overfitlab_study.settings import Settings

GRID = {"ma_cross": {"fast": [5, 10], "slow": [60, 120], "mode": ["long_short", "long_only"]}}
UNIVERSE = {"etfs": {"spy": "spy.us", "tlt": "tlt.us"}, "index": ["^ndx"], "fx": ["eurusd"]}


def business_days(count, start=dt.date(2015, 1, 2)):
    days = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += dt.timedelta(days=1)
    return days


def write_prices(path, days, seed):
    close = ol.gbm_prices(len(days), vol_annual=0.2, rng=seed)
    frame = pl.DataFrame({"date": days, "close": close})
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


@pytest.fixture
def processed(tmp_path):
    folder = tmp_path / "processed"
    days = business_days(1200)
    write_prices(tables.prices_path(folder, "sharadar", "spy"), days, 1)
    write_prices(tables.prices_path(folder, "stooq", "tlt.us"), days[5:], 2)
    write_prices(tables.prices_path(folder, "stooq", "^ndx"), days, 3)
    write_prices(tables.prices_path(folder, "stooq", "eurusd"), days, 4)
    rf = pl.DataFrame(
        {
            "date": days,
            "dataset": universe_module.RF_DATASET,
            "factor": "RF",
            "region": None,
            "frequency": "D",
            "ret": 0.0001,
        },
        schema=french.FACTOR_RETURNS_SCHEMA,
    )
    tables.write_table(rf, tables.factor_path(folder, "french", universe_module.RF_DATASET))
    return folder


def test_resolve_prefers_sharadar_and_falls_back_to_stooq(processed):
    series = universe_module.resolve(UNIVERSE, processed)
    names = {item.name: item for item in series}
    assert "sharadar" in str(names["spy"].path)
    assert "stooq" in str(names["tlt"].path)
    assert names["spy"].excess_of_rf
    assert not names["eurusd"].excess_of_rf


def test_excess_closes_take_off_the_risk_free_rate(processed):
    series = universe_module.resolve(UNIVERSE, processed)
    dates, closes, names = universe_module.aligned_excess_closes(series, processed)
    assert closes.shape == (dates.size, 4)
    assert (closes[0] == 1.0).all()
    assert names == ["spy", "tlt", "^ndx", "eurusd"]
    raw = pl.read_parquet(tables.prices_path(processed, "stooq", "eurusd"))
    joined = raw.filter(pl.col("date").is_in(dates.astype("datetime64[D]").tolist()))
    fx = joined.get_column("close").to_numpy()
    assert closes[-1, 3] == pytest.approx(fx[-1] / fx[0])
    ndx = pl.read_parquet(tables.prices_path(processed, "stooq", "^ndx"))
    ndx = ndx.filter(pl.col("date").is_in(dates.tolist())).get_column("close").to_numpy()
    assert closes[-1, 2] < ndx[-1] / ndx[0]


def test_holdout_start_is_the_last_calendar_years():
    assert stooq_factory.holdout_start(dt.date(2026, 10, 30), 3) == dt.date(2024, 1, 1)


def test_strategy_matrix_averages_the_series():
    closes = np.column_stack([ol.gbm_prices(100, rng=1), ol.gbm_prices(100, rng=2)])
    specs = ol.strategy_grid({"ma_cross": {"fast": [2], "slow": [5], "mode": ["long_short"]}})
    matrix = stooq_factory.strategy_matrix(closes, specs, 0.0)
    each = [ol.run_strategies(closes[:, k], specs, cost_bps=0.0) for k in range(2)]
    assert matrix == pytest.approx((each[0] + each[1]) / 2)


def run_case(processed, tmp_path):
    return stooq_factory.run(
        universe=UNIVERSE,
        grid=GRID,
        cost_bps=1.0,
        holdout_years=1,
        processed_dir=processed,
        results_dir=tmp_path / "results",
        reports_dir=tmp_path / "reports",
    )


def test_case_logs_every_trial_and_one_holdout(processed, tmp_path):
    row = run_case(processed, tmp_path)
    path = tmp_path / "results" / "trial_log.csv"
    assert [i.code for i in ol.validate_trial_log(path) if i.severity == "error"] == []
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 9
    assert [r["x_is_holdout"] for r in rows] == ["false"] * 8 + ["true"]
    assert rows[-1]["strategy_family"] == stooq_factory.HOLDOUT_FAMILY
    assert row["n_trials"] == 8
    assert row["end"] < row["holdout_start"]
    assert row["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE")
    assert (tmp_path / "reports/tables/case_stooq.csv").exists()


def test_rerun_never_evaluates_the_holdout_again(processed, tmp_path):
    first = run_case(processed, tmp_path)
    second = run_case(processed, tmp_path)
    assert first == second
    with (tmp_path / "results/trial_log.csv").open(encoding="utf-8", newline="") as fh:
        assert len(list(csv.DictReader(fh))) == 9


def write_factor(folder, source, dataset, columns, months=300, seed=0):
    rng = np.random.default_rng(seed)
    days = [tables.month_end(dt.date(2000 + i // 12, i % 12 + 1, 1)) for i in range(months)]
    records = [
        {
            "date": day,
            "dataset": dataset,
            "factor": name,
            "region": None,
            "frequency": "M",
            "ret": float(rng.normal(0.01, 0.03)),
        }
        for name in columns
        for day in days
    ]
    frame = pl.DataFrame(records, schema=french.FACTOR_RETURNS_SCHEMA)
    tables.write_table(frame, tables.factor_path(folder, source, dataset))


def test_factor_case(tmp_path):
    folder = tmp_path / "processed"
    write_factor(folder, "french", factors.FRENCH_MONTHLY, ["Mkt-RF", "SMB", "HML", "RF"])
    write_factor(folder, "aqr", "demo", ["A", "B"], seed=1)
    write_factor(folder, "aqr", "short", ["C"], months=100, seed=2)
    summary = factors.run(
        aqr_datasets=["demo", "short"], processed_dir=folder, reports_dir=tmp_path / "reports"
    )
    assert summary["factors"] == 5
    assert summary["dropped"] == ["aqr/short/C"]
    assert (tmp_path / "reports/tables/case_factors.csv").exists()


def test_bhy_by_hand():
    # c = 1 + 1/2 + 1/3 = 11/6: limits are 0.0091, 0.0182 and 0.0273 at q = 0.05.
    assert osap.bhy_rejections(np.array([0.005, 0.015, 0.9])) == 2
    assert osap.bhy_rejections(np.array([0.5, 0.6, 0.9])) == 0


def osap_tables(folder):
    rng = np.random.default_rng(0)
    days = [tables.month_end(dt.date(1990 + i // 12, i % 12 + 1, 1)) for i in range(360)]
    records = []
    for k, (name, drift) in enumerate((("Strong", 0.02), ("Weak", 0.0), ("Other", 0.001))):
        for day in days:
            records.append(
                {
                    "date": day,
                    "signal": name,
                    "portfolio": "LS",
                    "weighting": "op",
                    "ret": float(rng.normal(drift, 0.03)),
                    "sample_start": dt.date(1990, 1, 1) if k < 2 else None,
                    "sample_end": dt.date(2000, 12, 31) if k < 2 else None,
                }
            )
    frame = pl.DataFrame(records, schema=tables.SIGNAL_SCHEMA)
    tables.write_table(frame, tables.signal_path(folder, "demo"))
    docs = pl.DataFrame(
        {
            "signal": ["Strong", "Weak"],
            "start_year": [1990, 1990],
            "end_year": [2000, 2000],
            "published": [2002, 2002],
        }
    )
    tables.write_table(docs, folder / "signal_docs" / "release=demo" / "part-0.parquet")
    return frame, docs


def test_windows_follow_the_paper_dates(tmp_path):
    frame, docs = osap_tables(tmp_path)
    in_sample = osap.in_window(frame, "in_sample", docs)
    assert in_sample.get_column("date").dt.year().max() == 2000
    assert set(in_sample.get_column("signal")) == {"Strong", "Weak"}
    post = osap.in_window(frame, "post_publication", docs)
    assert post.get_column("date").dt.year().min() == 2003
    assert osap.in_window(frame, "all", docs).height == frame.height


def test_zero_filling_keeps_the_t_statistic():
    returns = np.random.default_rng(0).normal(0.01, 0.03, 240)
    filled = np.concatenate((returns, np.zeros(240)))
    t_own = returns.mean() / returns.std(ddof=1) * np.sqrt(returns.size)
    t_filled = filled.mean() / filled.std(ddof=1) * np.sqrt(filled.size)
    assert t_filled == pytest.approx(t_own, rel=0.05)


def test_osap_case(tmp_path):
    osap_tables(tmp_path / "processed")
    rows = osap.run(
        release="demo",
        processed_dir=tmp_path / "processed",
        reports_dir=tmp_path / "reports",
        n_boot=200,
    )
    assert [row["window"] for row in rows] == list(osap.WINDOWS)
    assert rows[0]["n_predictors"] == 3
    assert rows[0]["t_above_3"] >= 1
    assert rows[0]["after_stepm"] >= 1
    assert rows[0]["best_signal"] == "Strong"
    assert rows[1]["n_predictors"] == 2
    assert (tmp_path / "reports/figures/osap_hurdles.png").stat().st_size > 1000


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLAB_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("OLAB_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("OLAB_STAT_ARB_PAIRS_DIR", str(tmp_path / "pairs"))
    return Settings()


def config_file(tmp_path, families="null"):
    path = tmp_path / "projects.yaml"
    path.write_text(
        "projects:\n"
        "  - name: stat-arb-pairs\n"
        "    dir: ${OLAB_STAT_ARB_PAIRS_DIR}\n"
        "    trial_log: results/trial_log.csv\n"
        f"    families: {families}\n"
        "    selected: null\n",
        encoding="utf-8",
    )
    return path


def fill_log(tmp_path):
    folder = tmp_path / "pairs" / "results"
    trials = ol.TrialLogger(folder / "trial_log.csv", folder / "runs", repo_dir=tmp_path)
    days = np.datetime64("2020-01-01", "D") + np.arange(400)
    returns = np.random.default_rng(0).normal(0.0, 0.01, (400, 4))
    for n in range(4):
        trials.log(strategy_family="pairs", params={"n": n}, dates=days, returns=returns[:, n])
    return folder / "trial_log.csv"


def test_project_without_a_log(settings, tmp_path):
    [row] = projects.run(settings, config_file(tmp_path))
    assert (row["project"], row["status"]) == ("stat-arb-pairs", "no trial log")
    assert (tmp_path / "reports/tables/own_projects.csv").exists()
    assert not (tmp_path / "reports/figures/own_projects_naive_vs_deflated.png").exists()


def test_project_with_an_invalid_log(settings, tmp_path):
    log = fill_log(tmp_path)
    log.write_text(log.read_text("utf-8").replace(",pairs,", ",Bad-Family,"), encoding="utf-8")
    [row] = projects.run(settings, config_file(tmp_path))
    assert row["status"] == "E_FAMILY"


def test_project_with_a_valid_log(settings, tmp_path):
    fill_log(tmp_path)
    [row] = projects.run(settings, config_file(tmp_path))
    assert row["status"] == "valid"
    assert row["family"] == "pairs"
    assert row["n_raw"] == 4
    assert row["n_clustered"] == 4
    assert row["dsr"] is not None
    assert (tmp_path / "reports/figures/own_projects_naive_vs_deflated.png").exists()


def test_project_family_that_cannot_be_evaluated(settings, tmp_path):
    fill_log(tmp_path)
    [row] = projects.run(settings, config_file(tmp_path, "[missing]"))
    assert row["status"].startswith("not evaluated")


def test_projects_command(settings, tmp_path):
    result = CliRunner().invoke(app, ["projects", "--config", str(config_file(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "no trial log" in result.stdout
