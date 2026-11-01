"""Command-line interface ``overfitlab``."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import typer

import overfitlab as ol
from overfitlab.haircut import Method
from overfitlab.triallog import (
    add_manual_trial,
    evaluate_trial_log,
    read_wide_returns,
    validate_trial_log,
)

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
trials_app = typer.Typer(no_args_is_help=True, help="Trial-log commands (contract v1.0).")
app.add_typer(trials_app, name="trials")

JsonFlag = Annotated[bool, typer.Option("--json", help="Print one JSON object.")]
Skew = Annotated[float, typer.Option(help="Skewness.")]
Kurtosis = Annotated[float, typer.Option(help="Non-excess kurtosis (normal = 3).")]


def _print_version(value: bool) -> None:
    if value:
        typer.echo(ol.__version__)
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_print_version, is_eager=True, help="Print the version."
        ),
    ] = False,
) -> None:
    """Tell a real trading signal from a lucky one."""


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _emit(payload: Mapping[str, Any], as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(_clean(dict(payload)), sort_keys=True))
        return
    for key, value in payload.items():
        typer.echo(f"{key}: {value:.6g}" if isinstance(value, float) else f"{key}: {value}")


def _guarded(action: Callable[[], None]) -> None:
    try:
        action()
    except ValueError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error


def _per_period_sr(sr: float | None, sr_annual: float | None, periods: float | None) -> float:
    if sr is not None and sr_annual is None:
        return sr
    if sr_annual is not None and sr is None:
        if periods is None:
            raise ValueError("--sr-annual needs --periods-per-year")
        return sr_annual / math.sqrt(periods)
    raise ValueError("give exactly one of --sr and --sr-annual")


@app.command()
def psr(
    n_obs: Annotated[int, typer.Option(help="Number of observations.")],
    sr: Annotated[float | None, typer.Option(help="Per-period Sharpe ratio.")] = None,
    sr_annual: Annotated[float | None, typer.Option(help="Annual Sharpe ratio.")] = None,
    periods_per_year: Annotated[float | None, typer.Option(help="Periods per year.")] = None,
    skew: Skew = 0.0,
    kurtosis: Kurtosis = 3.0,
    benchmark: Annotated[float, typer.Option(help="Benchmark, same units as the SR.")] = 0.0,
    as_json: JsonFlag = False,
) -> None:
    """Probabilistic Sharpe Ratio."""

    def action() -> None:
        per_period = _per_period_sr(sr, sr_annual, periods_per_year)
        scale = 1.0 if sr is not None else math.sqrt(cast(float, periods_per_year))
        value = ol.probabilistic_sharpe_ratio(
            per_period, n_obs, sr_benchmark=benchmark / scale, skew=skew, kurtosis=kurtosis
        )
        std = ol.sharpe_std(per_period, n_obs, skew=skew, kurtosis=kurtosis)
        _emit({"psr": value, "sr_std": std}, as_json)

    _guarded(action)


@app.command()
def dsr(
    n_obs: Annotated[int, typer.Option(help="Number of observations.")],
    n_trials: Annotated[float, typer.Option(help="Number of independent trials.")],
    sr: Annotated[float | None, typer.Option(help="Per-period Sharpe ratio.")] = None,
    sr_annual: Annotated[float | None, typer.Option(help="Annual Sharpe ratio.")] = None,
    periods_per_year: Annotated[float | None, typer.Option(help="Periods per year.")] = None,
    skew: Skew = 0.0,
    kurtosis: Kurtosis = 3.0,
    var_trials: Annotated[float | None, typer.Option(help="Per-period V.")] = None,
    var_trials_annual: Annotated[float | None, typer.Option(help="Annual V.")] = None,
    as_json: JsonFlag = False,
) -> None:
    """Deflated Sharpe Ratio."""

    def action() -> None:
        per_period = _per_period_sr(sr, sr_annual, periods_per_year)
        if (var_trials is None) == (var_trials_annual is None):
            raise ValueError("give exactly one of --var-trials and --var-trials-annual")
        if var_trials_annual is not None and periods_per_year is None:
            raise ValueError("--var-trials-annual needs --periods-per-year")
        variance = (
            var_trials
            if var_trials is not None
            else cast(float, var_trials_annual) / cast(float, periods_per_year)
        )
        result = ol.deflated_sharpe_ratio(
            per_period, n_obs, n_trials=n_trials, var_trials=variance, skew=skew, kurtosis=kurtosis
        )
        payload: dict[str, Any] = {"sr0": result.sr0}
        if periods_per_year is not None:
            payload["sr0_annual"] = ol.annualize_sharpe(result.sr0, periods_per_year)
        _emit({**payload, "dsr": result.dsr, "z": result.z}, as_json)

    _guarded(action)


@app.command()
def mintrl(
    sr_annual: Annotated[float, typer.Option(help="Annual Sharpe ratio.")],
    periods_per_year: Annotated[float, typer.Option(help="Periods per year.")],
    skew: Skew = 0.0,
    kurtosis: Kurtosis = 3.0,
    benchmark_annual: Annotated[float, typer.Option(help="Annual benchmark.")] = 0.0,
    alpha: Annotated[float, typer.Option(help="Significance level.")] = 0.05,
    as_json: JsonFlag = False,
) -> None:
    """Minimum Track Record Length."""

    def action() -> None:
        root = math.sqrt(periods_per_year)
        observations = ol.min_track_record_length(
            sr_annual / root,
            sr_benchmark=benchmark_annual / root,
            skew=skew,
            kurtosis=kurtosis,
            alpha=alpha,
        )
        _emit({"observations": observations, "years": observations / periods_per_year}, as_json)

    _guarded(action)


@app.command()
def minbtl(
    n_trials: Annotated[float, typer.Option(help="Number of independent trials.")],
    max_sharpe_annual: Annotated[float, typer.Option(help="Target E[max SR].")] = 1.0,
    as_json: JsonFlag = False,
) -> None:
    """Minimum Backtest Length, in years."""
    _guarded(
        lambda: _emit(
            {"years": ol.min_backtest_length(n_trials, max_sharpe_annual=max_sharpe_annual)},
            as_json,
        )
    )


@app.command()
def haircut(
    sr_annual: Annotated[float, typer.Option(help="Annual Sharpe ratio.")],
    n_obs: Annotated[int, typer.Option(help="Number of observations.")],
    n_tests: Annotated[int, typer.Option(help="Number of tests.")],
    periods_per_year: Annotated[int, typer.Option(help="Periods per year.")] = 12,
    method: Annotated[str, typer.Option(help="bonferroni, holm or bhy.")] = "bhy",
    autocorr: Annotated[float, typer.Option(help="First-order autocorrelation.")] = 0.0,
    rho_tests: Annotated[float, typer.Option(help="Correlation among tests.")] = 0.2,
    seed: Annotated[int, typer.Option(help="Simulation seed.")] = 0,
    as_json: JsonFlag = False,
) -> None:
    """Harvey-Liu Sharpe ratio haircut."""

    def action() -> None:
        result = ol.haircut_sharpe(
            sr_annual,
            n_obs,
            periods_per_year=periods_per_year,
            n_tests=n_tests,
            method=cast(Method, method),
            autocorr=autocorr,
            rho_tests=rho_tests,
            rng=seed,
        )
        _emit(
            {
                "sr_annual": result.sr_annual,
                "haircut_sr_annual": result.haircut_sr_annual,
                "haircut_pct": result.haircut_pct,
                "tstat": result.tstat,
                "pvalue": result.pvalue,
                "adj_pvalue": result.adj_pvalue,
            },
            as_json,
        )

    _guarded(action)


@app.command()
def pbo(
    returns: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    partitions: Annotated[int, typer.Option(help="Number of CSCV blocks S, even.")] = 16,
    as_json: JsonFlag = False,
) -> None:
    """Probability of Backtest Overfitting of a wide return table."""

    def action() -> None:
        _, matrix = read_wide_returns(returns)
        result = ol.pbo_cscv(matrix, n_partitions=partitions)
        _emit(
            {
                "pbo": result.pbo,
                "degradation_slope": result.degradation_slope,
                "prob_oos_loss": result.prob_oos_loss,
                "n_combinations": result.n_combinations,
            },
            as_json,
        )

    _guarded(action)


@app.command()
def snoop(
    returns: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    benchmark_col: Annotated[str | None, typer.Option(help="Column of the benchmark.")] = None,
    zero_benchmark: Annotated[bool, typer.Option("--zero-benchmark")] = False,
    test: Annotated[str, typer.Option(help="rc, spa or stepm.")] = "spa",
    n_boot: Annotated[int, typer.Option(help="Resamples.")] = 1000,
    block_length: Annotated[str, typer.Option(help="Mean block, or auto.")] = "auto",
    seed: Annotated[int, typer.Option(help="Bootstrap seed.")] = 0,
    as_json: JsonFlag = False,
) -> None:
    """Data-snooping tests on a wide return table."""

    def action() -> None:
        if (benchmark_col is None) == (not zero_benchmark):
            raise ValueError("give exactly one of --benchmark-col and --zero-benchmark")
        names, matrix = read_wide_returns(returns)
        if benchmark_col is not None:
            if benchmark_col not in names:
                raise ValueError(f"no column {benchmark_col!r}")
            position = names.index(benchmark_col)
            kept = [i for i in range(len(names)) if i != position]
            matrix = matrix[:, kept] - matrix[:, [position]]
            names = tuple(names[i] for i in kept)
        block = None if block_length == "auto" else float(block_length)
        options: dict[str, Any] = {"n_boot": n_boot, "block_length": block, "rng": seed}
        if test == "rc":
            reality = ol.reality_check(matrix, **options)
            _emit({"stat": reality.stat, "pvalue": reality.pvalue}, as_json)
        elif test == "spa":
            spa = ol.spa_test(matrix, **options)
            _emit(
                {
                    "stat": spa.stat,
                    "pvalue_lower": spa.pvalue_lower,
                    "pvalue_consistent": spa.pvalue_consistent,
                    "pvalue_upper": spa.pvalue_upper,
                },
                as_json,
            )
        elif test == "stepm":
            stepm = ol.romano_wolf_stepdown(matrix, **options)
            rejected = [name for name, flag in zip(names, stepm.rejected, strict=True) if flag]
            _emit({"rejected": rejected, "n_steps": stepm.n_steps}, as_json)
        else:
            raise ValueError(f"unknown test {test!r}")

    _guarded(action)


@trials_app.command("validate")
def trials_validate(
    log_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    strict: Annotated[bool, typer.Option(help="Exit 1 on any error.")] = True,
    as_json: JsonFlag = False,
) -> None:
    """Check a trial log against contract v1.0."""
    issues = validate_trial_log(log_csv)
    errors = [issue for issue in issues if issue.severity == "error"]
    if as_json:
        records = [
            {"code": i.code, "severity": i.severity, "row": i.row, "message": i.message}
            for i in issues
        ]
        typer.echo(json.dumps({"n_errors": len(errors), "issues": records}, sort_keys=True))
    else:
        for issue in issues:
            location = "file" if issue.row is None else f"row {issue.row}"
            typer.echo(f"{issue.code} ({issue.severity}, {location}): {issue.message}")
        typer.echo(f"{len(errors)} errors, {len(issues) - len(errors)} warnings")
    if errors and strict:
        raise typer.Exit(code=1)


@trials_app.command("report")
def trials_report(
    log_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    family: Annotated[list[str] | None, typer.Option(help="Family in scope.")] = None,
    selected: Annotated[str | None, typer.Option(help="run_id of the selected trial.")] = None,
    n_effective: Annotated[str, typer.Option(help="raw or clustered.")] = "raw",
    exclude_failed: Annotated[bool, typer.Option("--exclude-failed")] = False,
    out: Annotated[Path | None, typer.Option(help="Write the JSON report here.")] = None,
    as_json: JsonFlag = False,
) -> None:
    """Evaluate a trial log: N, V, SR0, naive SR, DSR, PBO, haircut and verdict."""

    def action() -> None:
        report = evaluate_trial_log(
            log_csv,
            families=family,
            selected=selected,
            n_effective=cast(Literal["raw", "clustered"], n_effective),
            exclude_failed=exclude_failed,
        )
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(report.to_json() + "\n", encoding="utf-8")
        _emit(report.to_dict(), as_json)

    _guarded(action)


@trials_app.command("add-manual")
def trials_add_manual(
    log_csv: Path,
    family: Annotated[str, typer.Option(help="Strategy family.")],
    params_json: Annotated[str, typer.Option(help="All choices that define the trial.")],
    note: Annotated[str, typer.Option(help="What was tried.")],
    start: Annotated[str, typer.Option(help="First date, YYYY-MM-DD.")] = "",
    end: Annotated[str, typer.Option(help="Last date, YYYY-MM-DD.")] = "",
    sharpe_annual: Annotated[float | None, typer.Option(help="Sharpe ratio seen.")] = None,
    as_json: JsonFlag = False,
) -> None:
    """Log a hand iteration that left no saved return series."""

    def action() -> None:
        run_id = add_manual_trial(
            log_csv,
            strategy_family=family,
            params=json.loads(params_json),
            note=note,
            start=start,
            end=end,
            sharpe_annual=sharpe_annual,
        )
        _emit({"run_id": run_id}, as_json)

    _guarded(action)
