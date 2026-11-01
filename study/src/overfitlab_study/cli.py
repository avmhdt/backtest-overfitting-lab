"""Command-line interface ``olab-study``."""

from __future__ import annotations

from pathlib import Path

import httpx
import typer

from overfitlab_study.data.smoke import run_smoke
from overfitlab_study.explog import study_run
from overfitlab_study.settings import Settings
from overfitlab_study.validation import papers

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
data_app = typer.Typer(no_args_is_help=True, help="Ingest raw data to the schemas of the spec.")
app.add_typer(data_app, name="data")
validate_app = typer.Typer(no_args_is_help=True, help="Check against published numbers.")
app.add_typer(validate_app, name="validate")

GOLDEN_CASES = Path("tests/golden/paper_examples.yaml")


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


@validate_app.command("papers")
def validate_papers() -> None:
    """Compare every golden case with the value printed in its paper."""
    settings = Settings()
    target = settings.olab_reports_dir / "validation" / "paper_examples.csv"
    with study_run("validate papers", settings) as run:
        results = [papers.evaluate_case(case) for case in papers.load_cases(GOLDEN_CASES)]
        papers.write_report(results, target)
        sourced = [r for r in results if r.status == "sourced"]
        failed = [r for r in sourced if not r.passed]
        run.metrics.update(
            sourced=len(sourced),
            passed=len(sourced) - len(failed),
            pending=len(results) - len(sourced),
        )
    typer.echo(f"{len(sourced) - len(failed)} of {len(sourced)} sourced cases pass -> {target}")
    for result in failed:
        label = "known miss" if result.known_discrepancy else "miss"
        typer.echo(f"{label} {result.id}: printed {result.printed}, computed {result.computed}")
    for result in results:
        if result.status != "sourced":
            typer.echo(f"{result.status} {result.id}")
    if any(not result.known_discrepancy for result in failed):
        raise typer.Exit(code=1)
