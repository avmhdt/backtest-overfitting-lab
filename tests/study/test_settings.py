import sys
from pathlib import Path

import pytest

from overfitlab_study.settings import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("DATA_ROOT", "OLAB_DATA_DIR", "OLAB_SEED", "SHARADAR_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_defaults():
    settings = Settings()
    assert settings.data_dir == Path("data")
    assert settings.olab_seed == 20261030
    assert settings.sharadar_api_key is None


def test_data_dir_prefers_olab_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("OLAB_DATA_DIR", str(tmp_path / "explicit"))
    assert Settings().data_dir == tmp_path / "explicit"


def test_data_root_used_when_no_olab_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    assert Settings().raw_dir == tmp_path / "root" / "raw"


def test_empty_env_values_are_ignored(monkeypatch):
    monkeypatch.setenv("OLAB_DATA_DIR", "")
    monkeypatch.setenv("SHARADAR_API_KEY", "")
    settings = Settings()
    assert settings.data_dir == Path("data")
    assert settings.sharadar_api_key is None


def test_home_data_root_policy(monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "~/data/backtest-overfitting-lab")
    if sys.platform == "win32":
        assert Settings().data_dir == Path("data")
    else:
        assert Settings().data_dir == Path.home() / "data" / "backtest-overfitting-lab"


def test_key_is_not_printed(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "not-a-real-key")
    assert "not-a-real-key" not in repr(Settings())


def test_dotenv_is_read(tmp_path):
    (tmp_path / ".env").write_text("OLAB_SEED=7\n", encoding="utf-8")
    assert Settings().olab_seed == 7
