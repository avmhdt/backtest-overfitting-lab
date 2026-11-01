"""Study configuration from the environment and ``.env``."""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    sharadar_api_key: SecretStr | None = None
    tiingo_api_key: SecretStr | None = None
    stooq_dir: Path | None = None
    data_root: str | None = None
    olab_data_dir: str | None = None
    olab_results_dir: Path = Path("results")
    olab_reports_dir: Path = Path("reports")
    olab_seed: int = Field(default=20261030)
    olab_equity_factor_pipeline_dir: Path = Path("../equity-factor-pipeline")
    olab_llm_text_alpha_dir: Path = Path("../llm-text-alpha")
    olab_stat_arb_pairs_dir: Path = Path("../stat-arb-pairs")
    olab_regime_portfolio_construction_dir: Path = Path("../regime-portfolio-construction")
    olab_crypto_lob_alpha_dir: Path = Path("../crypto-lob-alpha")
    olab_market_making_sim_dir: Path = Path("../market-making-sim")

    @property
    def data_dir(self) -> Path:
        """Root of raw/, interim/ and processed/: OLAB_DATA_DIR, DATA_ROOT, then ./data."""
        if self.olab_data_dir:
            return Path(self.olab_data_dir).expanduser()
        if self.data_root and not self._is_windows_home_path(self.data_root):
            return Path(self.data_root).expanduser()
        return Path("data")

    @staticmethod
    def _is_windows_home_path(value: str) -> bool:
        # The .env.example default is a WSL path; on native Windows ~ would land on C:.
        return sys.platform == "win32" and value.startswith("~")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"
