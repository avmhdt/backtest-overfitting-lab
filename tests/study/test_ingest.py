import csv
import datetime as dt
import io
import json
import zipfile

import httpx
import openpyxl
import polars as pl
import pytest
from typer.testing import CliRunner

from overfitlab_study.cli import app
from overfitlab_study.data import aqr, french, osap, sharadar, stooq, tables
from overfitlab_study.settings import Settings

STOOQ_HEADER = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>"


def stooq_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [STOOQ_HEADER]
    lines += [f"X,D,{d},000000,{c},{c},{c},{c},{v},0" for d, c, v in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "stooq"
    stooq_file(
        root / "us" / "nyse etfs" / "2" / "spy.us.txt",
        [("20050225", 93.5, 100.5), ("20050228", 94.0, 0)],
    )
    stooq_file(root / "world" / "eurusd.txt", [("20050225", 1.3, 0), ("20050228", 1.31, 0)])
    return root


def ingest_stooq(snapshot, tmp_path, **changes):
    arguments = {
        "symbols": ["spy.us", "eurusd"],
        "adjusted": {"spy.us": True},
        "checks": [stooq.AdjustmentCheck("spy.us", dt.date(2005, 2, 25), 120.0)],
        "snapshot": "2026-10-30",
        "raw_dir": tmp_path / "raw" / "stooq",
        "interim_dir": tmp_path / "interim",
        "processed_dir": tmp_path / "processed",
        **changes,
    }
    return stooq.ingest(snapshot, **arguments)


def test_index_finds_files_in_nested_folders(snapshot):
    index = stooq.build_index(snapshot)
    assert index["spy.us"].name == "spy.us.txt"
    assert set(index) == {"spy.us", "eurusd"}


def test_index_is_cached(snapshot, tmp_path):
    cache = tmp_path / "cache" / "index.json"
    first = stooq.build_index(snapshot, cache)
    (snapshot / "world" / "eurusd.txt").unlink()
    assert stooq.build_index(snapshot, cache) == first


def test_parse_file_schema_and_units(snapshot):
    frame = stooq.parse_file(snapshot / "us/nyse etfs/2/spy.us.txt", "spy.us", adjusted=True)
    assert frame.schema == tables.PRICES_SCHEMA
    assert frame.get_column("date").to_list() == [dt.date(2005, 2, 25), dt.date(2005, 2, 28)]
    assert frame.get_column("volume").to_list() == [100.5, None]


def test_parse_file_needs_the_columns(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("<DATE>,<CLOSE>\n20050225,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        stooq.parse_file(path, "x", adjusted=False)


def test_adjustment_check():
    frame = pl.DataFrame({"date": [dt.date(2005, 2, 25)], "close": [93.5]})
    check = stooq.AdjustmentCheck("spy.us", dt.date(2005, 2, 25), 120.0)
    assert stooq.verify_adjustment(frame, check)
    assert not stooq.verify_adjustment(frame.with_columns(pl.lit(120.0).alias("close")), check)
    missing = stooq.AdjustmentCheck("spy.us", dt.date(2006, 1, 3), 120.0)
    with pytest.raises(ValueError, match="no bar"):
        stooq.verify_adjustment(frame, missing)


def test_stooq_ingest_writes_tables_and_manifest(snapshot, tmp_path):
    assert ingest_stooq(snapshot, tmp_path) == {"spy.us": 2, "eurusd": 2}
    written = pl.read_parquet(tables.prices_path(tmp_path / "processed", "stooq", "spy.us"))
    assert written.get_column("adjusted").all()
    manifest = json.loads((tmp_path / "raw/stooq/MANIFEST.json").read_text("utf-8"))
    assert manifest["notes"] == {
        "adjustment_verified": {"spy.us": True},
        "snapshot_date": "2026-10-30",
    }
    assert manifest["release"] == "2026-10-30"


def test_stooq_ingest_reports_missing_symbols(snapshot, tmp_path):
    with pytest.raises(ValueError, match="nope"):
        ingest_stooq(snapshot, tmp_path, symbols=["nope"])


def test_month_end():
    assert tables.month_end(dt.date(2024, 2, 3)) == dt.date(2024, 2, 29)


def test_extreme_returns_are_flagged_not_dropped():
    days = [dt.date(2024, 1, 1), dt.date(2024, 1, 2)]
    flags = tables.extreme_returns("x", days, [0.1, -0.6], tables.DAILY_FLAG)
    assert [(f["date"], f["value"]) for f in flags] == [("2024-01-02", -0.6)]


def test_null_returns_are_not_flagged():
    assert tables.extreme_returns("x", [dt.date(2024, 1, 1)], [None], 0.5) == []


def test_gaps_are_logged():
    days = [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 2, 1)]
    assert [f["value"] for f in tables.gaps("x", days)] == [30]


def test_flags_append_with_one_header(tmp_path):
    row = {"series": "x", "date": "2024-01-01", "kind": "gap_days", "value": 30}
    tables.append_flags(tmp_path, "src", [row])
    path = tables.append_flags(tmp_path, "src", [row])
    with path.open(encoding="utf-8", newline="") as fh:
        assert len(list(csv.DictReader(fh))) == 2


SFP_HEADER = "ticker,date,open,high,low,close,volume,closeadj,closeunadj"


def sfp_row(ticker, day, close=100.0, adj=50.0):
    return f"{ticker},{day},{close},{close},{close},{close},10,{adj},{close}"


def paged_client(rows, seen, *, trailing_newline):
    def handler(request):
        seen.append(request)
        skip = int(request.url.params["skip"])
        limit = int(request.url.params["limit"])
        body = "\n".join([SFP_HEADER, *rows[skip : skip + limit]])
        return httpx.Response(200, text=body + ("\n" if trailing_newline else ""))

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("trailing_newline", [True, False])
@pytest.mark.parametrize("count", [2, 3, 6, 7])
def test_fetch_reads_every_page(monkeypatch, count, trailing_newline):
    monkeypatch.setattr(sharadar, "PAGE_SIZE", 3)
    rows = [sfp_row("SPY", f"2024-01-{i + 1:02d}") for i in range(count)]
    with paged_client(rows, [], trailing_newline=trailing_newline) as client:
        text = sharadar.fetch(["SPY"], "k", client)
    assert list(csv.DictReader(io.StringIO(text)))[-1]["date"] == f"2024-01-{count:02d}"
    assert len(text.splitlines()) == count + 1


def test_fetch_sends_the_key_in_the_header_only():
    seen = []
    with paged_client([sfp_row("SPY", "2024-01-02")], seen, trailing_newline=True) as client:
        sharadar.fetch(["SPY"], "secret-value", client)
    assert seen[0].headers["x-api-key"] == "secret-value"
    assert "secret-value" not in str(seen[0].url)
    assert seen[0].url.params["from"] == sharadar.START


def test_prices_scale_the_bar_to_closeadj():
    text = "\n".join([SFP_HEADER, sfp_row("SPY", "2024-01-02", 100.0, 50.0)]) + "\n"
    frame = sharadar.to_prices(text)
    assert frame.schema == tables.PRICES_SCHEMA
    row = frame.row(0, named=True)
    assert (row["symbol"], row["open"], row["close"], row["adjusted"]) == ("spy", 50.0, 50.0, True)


def test_empty_pull_gives_an_empty_table():
    assert sharadar.to_prices(SFP_HEADER + "\n").is_empty()


def test_sharadar_adjustment_check():
    text = "\n".join([SFP_HEADER, sfp_row("SPY", "2005-02-25", 120.0, 90.0)]) + "\n"
    assert sharadar.verify_adjustment(text, "spy", dt.date(2005, 2, 25))
    with pytest.raises(ValueError, match="no bar"):
        sharadar.verify_adjustment(text, "spy", dt.date(2006, 1, 3))


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OLAB_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("OLAB_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)
    monkeypatch.delenv("STOOQ_DIR", raising=False)
    return Settings()


def test_missing_key_is_a_clear_error(settings):
    with pytest.raises(sharadar.MissingKeyError, match="Stooq"):
        sharadar.api_key(settings)


def test_sharadar_ingest_and_crosscheck(settings, monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "k")
    keyed = Settings()
    rows = [sfp_row("SPY", "2005-02-25", 120.0, 90.0), sfp_row("SPY", "2005-02-28", 121.0, 91.0)]
    rows.append(sfp_row("SPY", "2005-03-01", 122.0, 90.5))
    with paged_client(rows, [], trailing_newline=True) as client:
        counts = sharadar.ingest(
            ["spy"], keyed, client, check_symbol="spy", check_day=dt.date(2005, 2, 25)
        )
    assert counts == {"spy": 3}
    manifest = json.loads((keyed.raw_dir / "sharadar/MANIFEST.json").read_text("utf-8"))
    assert manifest["notes"] == {"adjustment_verified": True}

    target = tables.prices_path(keyed.processed_dir, "sharadar", "spy")
    copy = tables.prices_path(keyed.processed_dir, "stooq", "spy.us")
    copy.parent.mkdir(parents=True)
    copy.write_bytes(target.read_bytes())
    [row] = sharadar.crosscheck({"spy": "spy.us", "qqq": "qqq.us"}, keyed.processed_dir)
    assert row["return_correlation"] == pytest.approx(1.0)
    assert row["mean_abs_diff"] == 0.0


def workbook_with(path, rows, sheet="Data"):
    book = openpyxl.Workbook()
    ws = book.active
    ws.title = sheet
    for row in rows:
        ws.append(row)
    book.save(path)


def test_aqr_parses_a_header_with_a_blank_first_cell(tmp_path):
    path = tmp_path / "book.xlsx"
    workbook_with(
        path,
        [
            ["AQR notes"],
            [None, "TSMOM", "TSMOM^FX"],
            [dt.datetime(2020, 1, 15), 0.01, -0.02],
            [dt.datetime(2020, 2, 15), None, 0.03],
            ["Copyright text"],
            [dt.datetime(2020, 3, 15), 9.0, 9.0],
        ],
    )
    frame = aqr.parse_sheet(path, aqr.Workbook("demo", "book.xlsx", "Data", "M", None))
    assert frame.schema == french.FACTOR_RETURNS_SCHEMA
    assert frame.get_column("date").unique().sort().to_list() == [
        dt.date(2020, 1, 31),
        dt.date(2020, 2, 29),
    ]
    assert frame.filter(factor="TSMOM", date=dt.date(2020, 2, 29)).item(0, "ret") is None


def test_aqr_reads_text_dates_and_filters_columns(tmp_path):
    path = tmp_path / "book.xlsx"
    workbook_with(path, [["DATE", "USA", "Global", "AUS"], ["12/31/1930", 0.01, 0.02, "x"]], "BAB")
    workbook = aqr.Workbook("bab", "book.xlsx", "BAB", "M", ("USA", "Global"))
    frame = aqr.parse_sheet(path, workbook)
    assert sorted(frame.get_column("factor").to_list()) == ["Global", "USA"]


def test_aqr_reports_missing_columns(tmp_path):
    path = tmp_path / "book.xlsx"
    workbook_with(path, [["DATE", "USA"], ["12/31/1930", 0.01]])
    workbook = aqr.Workbook("x", "book.xlsx", "Data", "M", ("Nope",))
    with pytest.raises(ValueError, match="no columns"):
        aqr.parse_sheet(path, workbook)


def test_aqr_config_and_ingest(tmp_path):
    workbook_with(tmp_path / "book.xlsx", [["Date", "A"], ["01/31/2020", 2.5]])
    workbooks = aqr.workbooks_from_config(
        [{"dataset": "demo", "file": "book.xlsx", "sheet": "Data"}]
    )
    rows = aqr.ingest(
        workbooks,
        aqr_dir=tmp_path,
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
    )
    assert rows == {"demo": 1}
    assert (tmp_path / "MANIFEST.json").exists()
    assert "extreme_return" in (tmp_path / "interim/aqr/flags.csv").read_text("utf-8")


@pytest.fixture
def osap_dir(tmp_path):
    folder = tmp_path / "osap"
    folder.mkdir()
    pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 31), dt.date(2020, 2, 28)],
            "AM": [1.5, None],
            "BM": [-2.0, 150.0],
        }
    ).write_parquet(folder / osap.RETURNS_FILE)
    (folder / osap.DOC_FILE).write_text(
        "Acronym,Year,SampleStartYear,SampleEndYear\nAM,1992,1951,1979\n", encoding="utf-8"
    )
    return folder


def test_osap_long_table_in_decimals_at_month_end(osap_dir):
    frame = osap.to_signal_returns(
        pl.read_parquet(osap_dir / osap.RETURNS_FILE), osap.read_doc(osap_dir / osap.DOC_FILE)
    )
    assert frame.schema == tables.SIGNAL_SCHEMA
    assert frame.height == 3
    row = frame.filter(signal="AM").row(0, named=True)
    assert row["ret"] == pytest.approx(0.015)
    assert row["sample_end"] == dt.date(1979, 12, 31)
    assert row["weighting"] == "op"
    assert frame.filter(signal="BM").item(1, "date") == dt.date(2020, 2, 29)
    assert frame.filter(signal="BM").item(0, "sample_start") is None


def test_osap_ingest_records_the_release(osap_dir, tmp_path):
    rows = osap.ingest(
        release="2026.10",
        osap_dir=osap_dir,
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
    )
    assert rows == {"signal_returns": 3}
    assert tables.signal_path(tmp_path / "processed", "2026.10").exists()
    manifest = json.loads((osap_dir / "MANIFEST.json").read_text("utf-8"))
    assert manifest["release"] == "2026.10"
    assert "extreme_return" in (tmp_path / "interim/osap/flags.csv").read_text("utf-8")


def test_french_ingest_writes_each_dataset(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("x.csv", ",Mkt-RF\n20240102,   1.00\n")
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=buffer.getvalue()))
    )
    rows = french.ingest(
        ["demo"], client=client, raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed"
    )
    assert rows == {"demo": 1}
    assert tables.factor_path(tmp_path / "processed", "french", "demo").exists()


def test_stooq_command_needs_a_snapshot(settings):
    result = CliRunner().invoke(app, ["data", "stooq"])
    assert result.exit_code == 1


def test_sharadar_command_reports_the_missing_key(settings, monkeypatch):
    universe = {
        "etfs": {"spy": "spy.us"},
        "adjustment_checks": {"sharadar": {"symbol": "spy", "date": "2005-02-25"}},
    }
    monkeypatch.setattr("overfitlab_study.cli.load_universe", lambda: universe)
    result = CliRunner().invoke(app, ["data", "sharadar"])
    assert result.exit_code == 1
    assert "Stooq" in result.output
