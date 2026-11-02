"""Schema and data-quality checks on the real ingested tables. Local only."""

import json
from pathlib import Path

import polars as pl
import pytest

from overfitlab_study.cases.universe import load_universe
from overfitlab_study.data import french, tables
from overfitlab_study.settings import Settings

pytestmark = pytest.mark.data
REPO = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def processed():
    folder = REPO / Settings(_env_file=REPO / ".env").data_dir / "processed"
    if not folder.exists():
        pytest.skip("no ingested tables")
    return folder


@pytest.fixture(scope="module")
def universe():
    return load_universe(REPO / "experiments" / "universe.yaml")


def symbols(universe):
    yield from (("sharadar", ticker) for ticker in universe["etfs"])
    yield from (("stooq", ticker) for ticker in universe["etfs"].values())
    yield from (("stooq", ticker) for ticker in [*universe["fx"], *universe["index"]])


def test_prices_meet_the_schema_and_rules(processed, universe):
    for source, symbol in symbols(universe):
        frame = pl.read_parquet(tables.prices_path(processed, source, symbol))
        assert frame.schema == tables.PRICES_SCHEMA, symbol
        assert frame.get_column("date").is_sorted(), symbol
        assert frame.get_column("date").n_unique() == frame.height, symbol
        assert (frame.get_column("close") > 0).all(), symbol
        assert set(frame.get_column("source")) == {source}, symbol


def test_etf_closes_are_adjusted(processed, universe):
    for ticker in universe["etfs"]:
        frame = pl.read_parquet(tables.prices_path(processed, "sharadar", ticker))
        assert frame.get_column("adjusted").all(), ticker


def test_factor_tables_are_decimal_and_month_ended(processed):
    for path in (processed / "factor_returns").rglob("part-0.parquet"):
        frame = pl.read_parquet(path)
        assert frame.schema == french.FACTOR_RETURNS_SCHEMA, path
        assert frame.get_column("ret").drop_nulls().abs().median() < 0.2, path
        monthly = frame.filter(pl.col("frequency") == "M").get_column("date")
        assert (monthly == monthly.dt.month_end()).all(), path


def test_signal_returns_are_decimal(processed):
    frame = pl.read_parquet(tables.signal_path(processed, "2026.10"))
    assert frame.schema == tables.SIGNAL_SCHEMA
    assert frame.get_column("ret").abs().median() < 0.1
    assert frame.get_column("signal").n_unique() == 212
    assert frame.get_column("sample_end").null_count() == 0


def test_manifests_are_present_with_the_required_fields():
    raw = REPO / Settings(_env_file=REPO / ".env").data_dir / "raw"
    if not raw.exists():
        pytest.skip("no raw data")
    for source in ("french", "aqr", "osap", "stooq", "sharadar"):
        manifest = json.loads((raw / source / "MANIFEST.json").read_text("utf-8"))
        assert manifest["row_counts"], source
        assert manifest["units"], source
        for entry in manifest["files"].values():
            assert len(entry["sha256"]) == 64
            assert entry["downloaded_at_utc"]
    osap = json.loads((raw / "osap" / "MANIFEST.json").read_text("utf-8"))
    assert osap["release"] == "2026.10"
    assert json.loads((raw / "stooq" / "MANIFEST.json").read_text("utf-8"))["notes"][
        "adjustment_verified"
    ] == {"spy.us": True}
