"""Open Source Asset Pricing long-short returns to the signal_returns schema.

The release stores the original paper's portfolio for each predictor, so each row has that
paper's own weighting, not one scheme throughout. Monthly returns are published in percent.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from overfitlab_study.data import tables
from overfitlab_study.data.manifest import write_manifest

SOURCE = "osap"
RETURNS_FILE = "PredictorLSretWide.parquet"
DOC_FILE = "SignalDoc.csv"
WEIGHTING = "op"


def read_doc(path: Path) -> pl.DataFrame:
    """Sample years and publication year per predictor (DQ-11)."""
    doc = pl.read_csv(path, infer_schema_length=2000, ignore_errors=True)
    return doc.select(
        pl.col("Acronym").alias("signal"),
        pl.col("SampleStartYear").cast(pl.Int64).alias("start_year"),
        pl.col("SampleEndYear").cast(pl.Int64).alias("end_year"),
        pl.col("Year").cast(pl.Int64).alias("published"),
    )


def to_signal_returns(wide: pl.DataFrame, doc: pl.DataFrame) -> pl.DataFrame:
    """Long table in decimal returns, stamped at month end, with the sample window."""
    long = (
        wide.unpivot(index="date", variable_name="signal", value_name="percent")
        .drop_nulls("percent")
        .with_columns(
            pl.col("date").map_elements(tables.month_end, return_dtype=pl.Date),
            (pl.col("percent") / 100.0).alias("ret"),
            pl.lit("LS").alias("portfolio"),
            pl.lit(WEIGHTING).alias("weighting"),
        )
    )
    joined = long.join(doc, on="signal", how="left").with_columns(
        pl.col("start_year")
        .map_elements(lambda y: date(y, 1, 1), return_dtype=pl.Date)
        .alias("sample_start"),
        pl.col("end_year")
        .map_elements(lambda y: date(y, 12, 31), return_dtype=pl.Date)
        .alias("sample_end"),
    )
    return joined.select(list(tables.SIGNAL_SCHEMA)).sort("signal", "date")


def ingest(
    *,
    release: str,
    osap_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
) -> dict[str, int]:
    returns_path = osap_dir / RETURNS_FILE
    doc_path = osap_dir / DOC_FILE
    wide = pl.read_parquet(returns_path)
    doc = read_doc(doc_path)
    frame = to_signal_returns(wide, doc)
    tables.write_table(frame, tables.signal_path(processed_dir, release))
    tables.write_table(doc, processed_dir / "signal_docs" / f"release={release}" / "part-0.parquet")
    flags = tables.extreme_returns(
        "osap",
        frame.get_column("date").to_list(),
        frame.get_column("ret").to_list(),
        tables.MONTHLY_FLAG,
    )
    tables.append_flags(interim_dir, SOURCE, flags)
    write_manifest(
        osap_dir,
        source=SOURCE,
        files=[returns_path, doc_path],
        row_counts={"signal_returns": frame.height, "predictors": frame["signal"].n_unique()},
        units="percent in source, decimal in signal_returns",
        origin="huggingface.co/datasets/OpenSourceAP/CrossSection, Data-Release-2026.10",
        release=release,
        notes={"portfolio_set": "Full Sets OP", "weighting": "each original paper's own"},
    )
    return {"signal_returns": frame.height}
