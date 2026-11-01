"""`olab-study data smoke`: prove one download works and report what else is missing."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from overfitlab_study.data import french
from overfitlab_study.data.manifest import write_manifest
from overfitlab_study.settings import Settings

EXPECTED_FACTORS = {"Mkt-RF", "SMB", "HML", "RF"}


@dataclass(frozen=True)
class SmokeResult:
    french_rows: int
    french_last_date: str
    missing: tuple[str, ...]


def missing_inputs(settings: Settings) -> list[str]:
    """Presence checks only. None of these fail the smoke test."""
    missing: list[str] = []
    if settings.stooq_dir is None or not settings.stooq_dir.is_dir():
        missing.append(f"STOOQ_DIR not found: {settings.stooq_dir} (P-06)")
    if settings.sharadar_api_key is None:
        missing.append("SHARADAR_API_KEY not set: U.S. ETFs fall back to Stooq (P-17)")
    aqr = settings.raw_dir / "aqr"
    if not aqr.is_dir() or not any(aqr.glob("*.xlsx")):
        missing.append(f"AQR workbooks not found in {aqr} (P-07)")
    osap = settings.raw_dir / "osap"
    if not osap.is_dir() or not any(osap.iterdir()):
        missing.append(f"OSAP signal returns not found in {osap} (P-08)")
    return missing


def run_smoke(settings: Settings, client: httpx.Client) -> SmokeResult:
    raw_dir = settings.raw_dir / "french"
    dataset = french.SMOKE_DATASET
    path = french.download(dataset, raw_dir, client)
    table = french.to_factor_returns(dataset, french.read_zip_text(path))

    if table.schema != french.FACTOR_RETURNS_SCHEMA:
        raise ValueError(f"unexpected factor_returns schema: {table.schema}")
    factors = set(table.get_column("factor").unique())
    if factors != EXPECTED_FACTORS:
        raise ValueError(f"unexpected factors in {dataset}: {sorted(factors)}")
    if table.is_empty() or table.get_column("ret").abs().max() > 1.0:  # type: ignore[operator]
        raise ValueError(f"{dataset} looks wrong: returns not in decimal units")

    write_manifest(
        raw_dir,
        source="french",
        files=[path],
        row_counts={dataset: table.height},
        units="percent in source, decimal in factor_returns",
        origin=french.dataset_url(dataset),
    )
    return SmokeResult(
        french_rows=table.height,
        french_last_date=str(table.get_column("date").max()),
        missing=tuple(missing_inputs(settings)),
    )
