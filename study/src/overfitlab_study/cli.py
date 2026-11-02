"""Command-line interface ``olab-study``."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

import httpx
import typer

from overfitlab_study import bench as bench_module
from overfitlab_study.cases import factors as case_factors_module
from overfitlab_study.cases import osap as case_osap_module
from overfitlab_study.cases import projects as projects_module
from overfitlab_study.cases import stooq_factory
from overfitlab_study.cases.universe import load_universe
from overfitlab_study.config import load_study_config
from overfitlab_study.data import aqr as aqr_module
from overfitlab_study.data import french as french_module
from overfitlab_study.data import osap as osap_module
from overfitlab_study.data import sharadar as sharadar_module
from overfitlab_study.data import stooq as stooq_module
from overfitlab_study.data.smoke import run_smoke
from overfitlab_study.explog import config_hash, study_run
from overfitlab_study.reports import write_table as report_table
from overfitlab_study.settings import Settings
from overfitlab_study.sims import leakage, noise, power
from overfitlab_study.validation import arch_xcheck, papers, studentised_xcheck

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
data_app = typer.Typer(no_args_is_help=True, help="Ingest raw data to the schemas of the spec.")
app.add_typer(data_app, name="data")
validate_app = typer.Typer(no_args_is_help=True, help="Check against published numbers.")
app.add_typer(validate_app, name="validate")
sim_app = typer.Typer(no_args_is_help=True, help="Simulations with known ground truth.")
app.add_typer(sim_app, name="sim")
case_app = typer.Typer(no_args_is_help=True, help="Real-data case studies.")
app.add_typer(case_app, name="case")

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


@data_app.command("stooq")
def data_stooq(
    stooq_dir: Annotated[Path | None, typer.Option(help="Snapshot folder.")] = None,
) -> None:
    """Convert the universe from the owner's local Stooq snapshot."""
    settings = Settings()
    folder = stooq_dir or settings.stooq_dir
    if folder is None or not folder.is_dir():
        typer.echo(f"error: Stooq snapshot not found: {folder}", err=True)
        raise typer.Exit(code=1)
    universe = load_universe()
    etfs = list(universe["etfs"].values())
    symbols = [*etfs, *universe["fx"], *universe["index"]]
    item = universe["adjustment_checks"]["stooq"]
    check = stooq_module.AdjustmentCheck(
        item["symbol"], date.fromisoformat(item["date"]), item["unadjusted_close"]
    )
    with study_run("data stooq", settings, params={"symbols": symbols}) as run:
        rows = stooq_module.ingest(
            folder,
            symbols=symbols,
            adjusted=dict.fromkeys(etfs, True),
            checks=[check],
            snapshot=universe["stooq_snapshot"],
            raw_dir=settings.raw_dir / "stooq",
            interim_dir=settings.interim_dir,
            processed_dir=settings.processed_dir,
        )
        run.metrics.update(rows=sum(rows.values()))
    typer.echo(f"stooq: {sum(rows.values())} rows for {len(rows)} symbols")


@data_app.command("sharadar")
def data_sharadar() -> None:
    """Pull U.S. ETF prices from Sharadar SFP."""
    settings = Settings()
    universe = load_universe()
    symbols = list(universe["etfs"])
    item = universe["adjustment_checks"]["sharadar"]
    try:
        with study_run("data sharadar", settings, params={"symbols": symbols}) as run:
            with httpx.Client() as client:
                rows = sharadar_module.ingest(
                    symbols,
                    settings,
                    client,
                    check_symbol=item["symbol"],
                    check_day=date.fromisoformat(item["date"]),
                )
            run.metrics.update(rows=sum(rows.values()))
    except sharadar_module.MissingKeyError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    agreement = sharadar_module.crosscheck(universe["etfs"], settings.processed_dir)
    if agreement:
        report_table(agreement, settings.olab_reports_dir / "tables" / "sharadar_stooq_xcheck.csv")
    typer.echo(f"sharadar: {sum(rows.values())} rows for {len(rows)} symbols")


@data_app.command("french")
def data_french() -> None:
    """Download the Ken French datasets of the universe."""
    settings = Settings()
    datasets = load_universe()["french"]["datasets"]
    with study_run("data french", settings, params={"datasets": datasets}) as run:
        with httpx.Client() as client:
            rows = french_module.ingest(
                datasets,
                client=client,
                raw_dir=settings.raw_dir / "french",
                processed_dir=settings.processed_dir,
            )
        run.metrics.update(rows)
    typer.echo(f"french: {rows}")


@data_app.command("aqr")
def data_aqr(
    folder: Annotated[Path | None, typer.Option("--dir", help="Workbook folder.")] = None,
) -> None:
    """Parse the AQR workbooks named in the universe."""
    settings = Settings()
    workbooks = aqr_module.workbooks_from_config(load_universe()["aqr"])
    with study_run("data aqr", settings) as run:
        rows = aqr_module.ingest(
            workbooks,
            aqr_dir=folder or settings.raw_dir / "aqr",
            interim_dir=settings.interim_dir,
            processed_dir=settings.processed_dir,
        )
        run.metrics.update(rows)
    typer.echo(f"aqr: {rows}")


@data_app.command("osap")
def data_osap(
    release: Annotated[str | None, typer.Option(help="OSAP release.")] = None,
) -> None:
    """Parse the owner's Open Source Asset Pricing files."""
    settings = Settings()
    version = release or load_study_config()["cases"]["osap"]["release"]
    with study_run("data osap", settings, params={"release": version}) as run:
        rows = osap_module.ingest(
            release=version,
            osap_dir=settings.raw_dir / "osap",
            interim_dir=settings.interim_dir,
            processed_dir=settings.processed_dir,
        )
        run.metrics.update(rows)
    typer.echo(f"osap: {rows}")


@case_app.command("stooq")
def case_stooq() -> None:
    """Run the factory on the universe, with logged trials and one holdout."""
    settings = Settings()
    study = load_study_config()
    section = study["cases"]["stooq"]
    with study_run("case stooq", settings, params=section) as run:
        row = stooq_factory.run(
            universe=load_universe(),
            grid=study["factory"]["grid"],
            cost_bps=study["factory"]["cost_bps"],
            holdout_years=section["holdout_years"],
            processed_dir=settings.processed_dir,
            results_dir=settings.olab_results_dir,
            reports_dir=settings.olab_reports_dir,
        )
        run.metrics.update(row)
    typer.echo(f"case stooq: {row}")


@case_app.command("factors")
def case_factors() -> None:
    """StepM survivors and haircuts across French and AQR factors."""
    settings = Settings()
    datasets = [item["dataset"] for item in load_universe()["aqr"]]
    with study_run("case factors", settings, params={"aqr": datasets}) as run:
        summary = case_factors_module.run(
            aqr_datasets=datasets,
            processed_dir=settings.processed_dir,
            reports_dir=settings.olab_reports_dir,
        )
        run.metrics.update(summary)
    typer.echo(f"case factors: {summary}")


@case_app.command("osap")
def case_osap() -> None:
    """Count the published predictors that survive multiple-testing corrections."""
    settings = Settings()
    release = load_study_config()["cases"]["osap"]["release"]
    with study_run("case osap", settings, params={"release": release}) as run:
        rows = case_osap_module.run(
            release=release,
            processed_dir=settings.processed_dir,
            reports_dir=settings.olab_reports_dir,
        )
        run.metrics.update(rows[0])
    for row in rows:
        typer.echo(f"case osap: {row}")


@app.command()
def projects(
    config: Annotated[Path, typer.Option(help="Project list.")] = projects_module.PROJECTS,
) -> None:
    """Write the naive against deflated table for the owner's own projects."""
    settings = Settings()
    with study_run("projects", settings, params={"config": str(config)}) as run:
        rows = projects_module.run(settings, config)
        run.metrics.update(evaluated=sum(row["dsr"] is not None for row in rows))
    for row in rows:
        typer.echo(f"{row['project']}: {row['status']}")


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


@sim_app.command("noise")
def sim_noise(
    seeds: Annotated[int | None, typer.Option(help="Replications; default from config.")] = None,
    quick: Annotated[bool, typer.Option("--quick", help="Reduced size for CI.")] = False,
) -> None:
    """Measure what the tests say about the best of many strategies on pure noise."""
    settings = Settings()
    study = load_study_config()
    config = noise.NoiseConfig.from_mapping(study["noise"], study["factory"]["cost_bps"], quick)
    count = config.seeds if seeds is None else seeds
    params = {"seeds": count, "quick": quick, "master_seed": settings.olab_seed}
    with study_run("sim noise", settings, params=params, config=study["noise"]) as run:
        rows, criteria = noise.run(
            config,
            seeds=count,
            master_seed=settings.olab_seed,
            reports_dir=settings.olab_reports_dir,
            results_dir=settings.olab_results_dir,
            config_key=config_hash({**study["noise"], **params}),
        )
        run.metrics.update(criteria)
    for row in rows:
        typer.echo(
            f"{row['setting']}: best SR {row['median_best_sr_annual']:.3f}, "
            f"DSR<0.95 in {row['seeds_dsr_below_95']}/{row['n_seeds']}, "
            f"median PBO {row['median_pbo']:.3f}, PASS {row['seeds_pass']}"
        )
    typer.echo(f"AC-W-01: {criteria}")


@validate_app.command("studentised")
def validate_studentised(
    n_boot: Annotated[int, typer.Option(help="Resamples per case.")] = 10_000,
) -> None:
    """Cross-check against bsds.m and StepwiseTest, with Octave and R in WSL."""
    settings = Settings()
    target = settings.olab_reports_dir / "validation" / "studentised_xcheck.csv"
    with study_run("validate studentised", settings, params={"n_boot": n_boot}) as run:
        rows = studentised_xcheck.run_all(
            n_boot,
            root=settings.interim_dir / "xcheck",
            toolbox=settings.data_dir / "external" / "mfe-toolbox",
        )
        studentised_xcheck.write_report(rows, target)
        criteria = studentised_xcheck.check_criteria(rows)
        run.metrics.update(criteria)
    typer.echo(f"AC-W-07: {criteria} -> {target}")


@sim_app.command("power")
def sim_power(
    check: Annotated[bool, typer.Option("--check", help="Only the cells of AC-W-02.")] = False,
    quick: Annotated[bool, typer.Option("--quick", help="Reduced size for CI.")] = False,
) -> None:
    """Measure how much signal and history each test needs."""
    settings = Settings()
    study = load_study_config()
    config = power.PowerConfig.from_mapping(study, quick)
    params = {"check": check, "quick": quick, "master_seed": settings.olab_seed}
    with study_run("sim power", settings, params=params, config=study["power"]) as run:
        if check:
            rows, criteria = power.run_check(
                config,
                master_seed=settings.olab_seed,
                reports_dir=settings.olab_reports_dir,
                quick=quick,
            )
        else:
            rows, criteria = power.run_grid(
                config,
                master_seed=settings.olab_seed,
                reports_dir=settings.olab_reports_dir,
                docs_dir=Path("docs"),
            )
        run.metrics.update(criteria)
    typer.echo(f"{len(rows)} rows: {criteria}")
    if check:
        for row in rows:
            if not row["within_tolerance"]:
                typer.echo(
                    f"outside: {row['test']} SR {row['true_sharpe_annual']} "
                    f"{row['years']}y N={row['n_trials']}: {row['empirical']:.3f} "
                    f"vs {row['reference']:.3f}"
                )


@validate_app.command("arch")
def validate_arch(
    n_boot: Annotated[int, typer.Option(help="Resamples per case.")] = 10_000,
) -> None:
    """Cross-check the snooping tests against the independent arch package."""
    settings = Settings()
    target = settings.olab_reports_dir / "validation" / "arch_xcheck.csv"
    with study_run("validate arch", settings, params={"n_boot": n_boot}) as run:
        rows = arch_xcheck.run_all(n_boot)
        arch_xcheck.write_report(rows, target)
        criteria = arch_xcheck.check_criteria(rows)
        run.metrics.update(criteria)
    typer.echo(f"arch evidence (AC-W-07 stays open): {criteria} -> {target}")
    if not arch_xcheck.passed(criteria):
        raise typer.Exit(code=1)


@app.command()
def bench() -> None:
    """Measure the performance requirements on this machine."""
    settings = Settings()
    with study_run("bench", settings) as run:
        rows = bench_module.run(settings.olab_reports_dir)
        run.metrics.update({row["case"]: row["seconds"] for row in rows})
    for row in rows:
        typer.echo(f"{row['requirement']} {row['case']}: {row['seconds']:.3f} s, met={row['met']}")
