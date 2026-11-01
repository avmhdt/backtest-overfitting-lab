"""Command-line interface ``olab-study``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import httpx
import typer

from overfitlab_study import bench as bench_module
from overfitlab_study.config import load_study_config
from overfitlab_study.data.smoke import run_smoke
from overfitlab_study.explog import config_hash, study_run
from overfitlab_study.settings import Settings
from overfitlab_study.sims import leakage
from overfitlab_study.validation import papers

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
data_app = typer.Typer(no_args_is_help=True, help="Ingest raw data to the schemas of the spec.")
app.add_typer(data_app, name="data")
validate_app = typer.Typer(no_args_is_help=True, help="Check against published numbers.")
app.add_typer(validate_app, name="validate")
sim_app = typer.Typer(no_args_is_help=True, help="Simulations with known ground truth.")
app.add_typer(sim_app, name="sim")

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


@sim_app.command("leakage")
def sim_leakage(
    seeds: Annotated[int | None, typer.Option(help="Replications; default from config.")] = None,
) -> None:
    """Compare standard, purged and combinatorial k-fold on overlapping labels."""
    settings = Settings()
    section = load_study_config()["leakage"]
    count = section["seeds"] if seeds is None else seeds
    params = {**section, "seeds": count, "master_seed": settings.olab_seed}
    with study_run("sim leakage", settings, params=params) as run:
        rows, criteria = leakage.run(
            leakage.LeakageConfig.from_mapping(section),
            seeds=count,
            master_seed=settings.olab_seed,
            reports_dir=settings.olab_reports_dir,
            results_dir=settings.olab_results_dir,
            config_key=config_hash(params),
        )
        run.metrics.update({row["row"]: row["mean"] for row in rows}, **criteria)
    for row in rows:
        typer.echo(f"{row['row']}: {row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
    typer.echo(f"AC-W-04: {criteria}")


@app.command()
def bench() -> None:
    """Measure the performance requirements on this machine."""
    settings = Settings()
    with study_run("bench", settings) as run:
        rows = bench_module.run(settings.olab_reports_dir)
        run.metrics.update({row["case"]: row["seconds"] for row in rows})
    for row in rows:
        typer.echo(f"{row['requirement']} {row['case']}: {row['seconds']:.3f} s, met={row['met']}")
