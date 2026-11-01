import datetime as dt
import io
import json
import zipfile

import httpx
import pytest
from typer.testing import CliRunner

from overfitlab_study.cli import app
from overfitlab_study.data import french
from overfitlab_study.data.smoke import missing_inputs, run_smoke
from overfitlab_study.settings import Settings

SAMPLE = """This file was created using the 202609 CRSP database.
The 1-month TBill rate data are provided by Ibbotson.

,Mkt-RF,SMB,HML,RF
192607,    2.89,   -2.55,   -2.39,    0.22
192608,    2.64,   -1.14,    3.81,    0.25
192609,  -99.99,   -1.36,    0.05,    0.23

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
1927,   29.47,   -2.46,   -3.75,    3.12

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


@pytest.fixture
def zipped():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("F-F_Research_Data_Factors.csv", SAMPLE)
    return buffer.getvalue()


def test_parse_tables_splits_monthly_and_annual():
    tables = french.parse_tables(SAMPLE)
    assert [len(table.rows) for table in tables] == [3, 1]
    assert tables[0].columns == ("Mkt-RF", "SMB", "HML", "RF")


def test_percent_to_decimal_and_month_end_stamp():
    table = french.to_factor_returns("demo", SAMPLE)
    row = table.filter(factor="Mkt-RF", date=dt.date(1926, 7, 31)).row(0, named=True)
    assert row["ret"] == pytest.approx(0.0289)
    assert row["frequency"] == "M"


def test_sentinel_becomes_null():
    table = french.to_factor_returns("demo", SAMPLE)
    assert table.filter(factor="Mkt-RF", date=dt.date(1926, 9, 30)).item(0, "ret") is None


def test_annual_table_is_dropped():
    assert french.to_factor_returns("demo", SAMPLE).get_column("date").dt.year().max() == 1926


def test_daily_dates_kept():
    text = ",A\n19260701,  0.10\n19260702,  0.20\n"
    table = french.to_factor_returns("demo", text)
    assert table.get_column("date").to_list() == [dt.date(1926, 7, 1), dt.date(1926, 7, 2)]
    assert table.get_column("frequency").unique().to_list() == ["D"]


def test_schema_matches_factor_returns():
    assert french.to_factor_returns("demo", SAMPLE).schema == french.FACTOR_RETURNS_SCHEMA


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OLAB_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)
    monkeypatch.delenv("STOOQ_DIR", raising=False)
    return Settings()


def mock_client(body: bytes, status: int = 200) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, content=body))
    )


def test_smoke_writes_manifest(settings, zipped):
    result = run_smoke(settings, mock_client(zipped))
    assert result.french_rows == 12
    assert result.french_last_date == "1926-09-30"
    manifest = json.loads((settings.raw_dir / "french" / "MANIFEST.json").read_text("utf-8"))
    entry = manifest["files"]["F-F_Research_Data_Factors_CSV.zip"]
    assert len(entry["sha256"]) == 64
    assert entry["origin"].endswith("F-F_Research_Data_Factors_CSV.zip")
    assert manifest["row_counts"] == {"F-F_Research_Data_Factors": 12}


def test_smoke_lists_missing_inputs(settings, zipped):
    missing = run_smoke(settings, mock_client(zipped)).missing
    assert len(missing) == 4
    assert any("SHARADAR_API_KEY" in item for item in missing)


def test_nothing_missing_when_all_present(tmp_path, monkeypatch, settings):
    stooq = tmp_path / "stooq"
    stooq.mkdir()
    for folder, name in (("aqr", "a.xlsx"), ("osap", "b.csv")):
        (settings.raw_dir / folder).mkdir(parents=True)
        (settings.raw_dir / folder / name).write_text("x", encoding="utf-8")
    monkeypatch.setenv("STOOQ_DIR", str(stooq))
    monkeypatch.setenv("SHARADAR_API_KEY", "not-a-real-key")
    assert missing_inputs(Settings()) == []


def test_smoke_fails_on_http_error(settings):
    with pytest.raises(httpx.HTTPStatusError):
        run_smoke(settings, mock_client(b"", status=503))


def test_smoke_fails_on_wrong_factors(settings):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("x.csv", ",Other\n192607,  1.00\n")
    with pytest.raises(ValueError, match="unexpected factors"):
        run_smoke(settings, mock_client(buffer.getvalue()))


def test_smoke_cli_exit_code_and_log(settings, zipped, monkeypatch):
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=zipped))
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda: real_client(transport=transport))
    result = CliRunner().invoke(app, ["data", "smoke"])
    assert result.exit_code == 0, result.output
    assert "french: 12 rows" in result.stdout
    assert (settings.olab_results_dir / "experiment_log.csv").exists()
