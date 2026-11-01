import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "check_hygiene", Path(__file__).parents[2] / "scripts" / "check_hygiene.py"
)
assert spec is not None and spec.loader is not None
hygiene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hygiene)


def test_clean_paths_pass():
    assert hygiene.find_problems(["src/a.py", "tests/fixtures/x.parquet"], {}) == []


def test_data_directory_blocked():
    assert "tracked file under data/" in hygiene.find_problems(["data/raw/x.csv"], {})[0]


def test_licensed_extension_blocked_outside_fixtures():
    assert hygiene.find_problems(["reports/x.xlsx"], {})


def test_large_file_blocked():
    assert hygiene.find_problems(["reports/x.csv"], {"reports/x.csv": 2_000_000})
