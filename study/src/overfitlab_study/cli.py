"""Command-line interface ``olab-study``."""

from __future__ import annotations

import httpx
import typer

from overfitlab_study.data.smoke import run_smoke
from overfitlab_study.explog import study_run
from overfitlab_study.settings import Settings

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
data_app = typer.Typer(no_args_is_help=True, help="Ingest raw data to the schemas of the spec.")
app.add_typer(data_app, name="data")


@app.callback()
def main() -> None:
    """Regenerate every table and figure of the study from data and seeds."""


@data_app.command("smoke")
def data_smoke() -> None:
    """Download one Ken French file, parse it and report which other inputs are missing."""
    settings = Settings()
    with study_run("data smoke", settings) as run, httpx.Client() as client:
        result = run_smoke(settings, client)
        run.metrics.update(
            french_rows=result.french_rows,
            french_last_date=result.french_last_date,
            n_missing=len(result.missing),
        )
    typer.echo(f"french: {result.french_rows} rows through {result.french_last_date}")
    for item in result.missing:
        typer.echo(f"missing: {item}")
