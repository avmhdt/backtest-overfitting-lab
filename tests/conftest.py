import csv
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def build_parquet(series_csv: Path, target: Path) -> None:
    with series_csv.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    dates = pa.array([row[0] for row in body]).cast(pa.date32())
    values = pa.array([float(row[1]) if row[1] else None for row in body], type=pa.float64())
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({header[0]: dates, header[1]: values}), target)


def materialise_fixture(name: str, destination: Path) -> Path:
    """Copy a CSV fixture and build its Parquet files; returns the log path."""
    source = FIXTURES / name
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy(source / "trial_log.csv", destination / "trial_log.csv")
    for series_csv in (
        sorted((source / "series").glob("*.csv")) if (source / "series").exists() else []
    ):
        build_parquet(series_csv, destination / "runs" / series_csv.stem / "returns.parquet")
    return destination / "trial_log.csv"


@pytest.fixture
def materialise(tmp_path):
    return lambda name: materialise_fixture(name, tmp_path / name.replace("/", "_"))
