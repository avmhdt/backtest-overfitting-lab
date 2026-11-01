# Trial-log contract v1.0

**Status: frozen 2026-11-01 (decision D-11).** Later changes only add optional `x_` columns, with the owner's approval and a minor version bump.

`equity-factor-pipeline`, `llm-text-alpha`, `stat-arb-pairs`, `regime-portfolio-construction`, `crypto-lob-alpha` and `market-making-sim` emit this format, and `overfitlab` reads it. `crypto-lob-alpha` and `market-making-sim` set `x_periods_per_year = 365` on their daily crypto return series.

A machine-readable description lives in [`src/overfitlab/trial_log_schema.json`](https://github.com/avmhdt/backtest-overfitting-lab/blob/main/src/overfitlab/trial_log_schema.json), which sibling repos may vendor.

## Files and location

- **Log:** `results/trial_log.csv` in the emitting repo, committed. One row per evaluated trial, appended when the evaluation finishes. Append-only: rows are never edited or deleted.
- **Return series:** one Parquet file per trial at `results/runs/<run_id>/returns.parquet`, git-ignored by `*.parquet` and regenerable by the emitting repo. The run directory may hold other per-run artifacts, which readers ignore.
- **CSV dialect:** UTF-8 without BOM, comma separator, header row, RFC 4180 quoting (fields containing commas or quotes, such as `params_json`, are double-quoted with inner quotes doubled), LF line endings, no comment lines, no index column.

## Columns

Exact names, exact order.

| Column | Type | Format and rule | Meaning |
|---|---|---|---|
| `run_id` | str | UUID v4, lower case canonical form, unique in the file | Trial identifier, and the name of the run directory `runs/<run_id>/` |
| `timestamp` | str | ISO 8601 UTC, second precision, `YYYY-MM-DDTHH:MM:SSZ` | When the evaluation finished |
| `git_sha` | str | regex `^[0-9a-f]{40}(-dirty)?$`; `-dirty` if the working tree had uncommitted changes | Code version that produced the trial |
| `strategy_family` | str | regex `^[a-z][a-z0-9_]{0,63}$` | Group of related trials, the unit for N, V and PBO |
| `params_json` | str | Canonical JSON object: `json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)`; JSON types only | All choices that define the trial: parameters, universe, feature set, model, cost assumption |
| `start` | str | `YYYY-MM-DD`; equals the first date in the Parquet | First return date evaluated |
| `end` | str | `YYYY-MM-DD`; equals the last date; `start` ≤ `end` | Last return date evaluated |
| `n_obs` | int | ≥ 0; equals the Parquet row count; 0 for failed trials | Number of return observations |
| `sharpe_annual` | float | Decimal with at least 6 significant digits; empty for failed trials | `mean(ret)/std(ret, ddof=1)·√q`, with q = `x_periods_per_year` when present, else 252 daily, 52 weekly, 12 monthly |
| `returns_path` | str | POSIX path relative to the CSV's directory, that is `runs/<run_id>/returns.parquet`; empty for failed trials | Location of the return series |
| `x_periods_per_year` (optional) | int | Optional v1.0 extension, placed among the `x_` columns after the ten required ones; positive integer such as 252, 365, 52 or 12, else `E_PERIODS`; one value per `strategy_family` | Periods per year q of this trial's return series. When present, readers use it instead of inferring q from date spacing (24/7 markets: 365 for daily crypto returns) |

Optional extra columns are allowed only after the ten required ones and only with the prefix `x_`: for example `x_selected` (`true` or `false`, marking the trial the project reports as its headline), `x_note`, `x_dataset`, `x_is_holdout`. Readers ignore unknown `x_` columns. Any other extra column is an error.

Rows with `n_obs = 0` may leave `start` and `end` empty. An empty `x_periods_per_year` cell means the same as the column being absent for that row.

## Parquet return series

- At least two columns: `date` (date32; a timestamp is accepted and cast if every value is midnight) and `ret` (float64). Additional columns, such as `ret_gross`, are allowed and ignored by readers. `E_RETURNS_SCHEMA` fires only when `date` or `ret` is missing or has the wrong type.
- `ret` is the simple net return of the strategy per period, after costs, in decimal, over the window used to judge the trial. For CV-based trials that is the concatenated out-of-sample predictions' returns. Returns may be raw or excess: the emitter records which in `x_returns_basis` (`excess` or `raw`), and raw is assumed by default.
- Dates are strictly increasing, with no duplicates, no nulls and no NaN or inf. Flat periods are 0.0, not missing.

## Emitter rules

These are what make the count honest.

- Log every evaluation whose result a human or an automated search looked at: grid points, random-search draws, hand edits re-run from a notebook, and the final holdout evaluation. Logging happens inside the shared backtest or evaluation function, so calling code cannot skip it.
- Failed or aborted runs are logged with `n_obs = 0`, empty `sharpe_annual` and empty `returns_path`. They count in N.
- Ideas tried by hand without a saved series are logged with `overfitlab trials add-manual`.
- Re-running the identical trial (same `strategy_family`, `params_json`, `start`, `end`, `git_sha`) appends a new row. Readers deduplicate on that key only when `mode="dedup"` is requested.
- Sibling repos either depend on `overfitlab` and use `TrialLogger`, or vendor the reference writer below. Both must pass `overfitlab trials validate results/trial_log.csv` in their CI. A CI step that skips with a notice when `overfitlab` is not installed is acceptable.

## Validation codes

**Errors:**

| Code | Meaning |
|---|---|
| `E_COLUMNS` | Missing, misordered or unknown non-`x_` column |
| `E_RUN_ID` | Bad format or duplicate |
| `E_TIMESTAMP` | Bad timestamp |
| `E_GIT_SHA` | Bad commit hash |
| `E_FAMILY` | Bad family name |
| `E_PARAMS_JSON` | Invalid or non-canonical JSON |
| `E_DATES` | Bad format, or `start` after `end` |
| `E_NOBS` | Bad `n_obs`, or a mismatch with the Parquet |
| `E_SHARPE` | `sharpe_annual` is not a finite decimal, or is empty on a row with a series |
| `E_RETURNS_MISSING` | The Parquet is missing while others exist, or `returns_path` is not the required path |
| `E_RETURNS_SCHEMA` | `date` or `ret` is missing or has the wrong type |
| `E_RETURNS_VALUES` | Null, NaN or inf |
| `E_RETURNS_ORDER` | Unsorted or duplicate dates |
| `E_WINDOW` | `start` or `end` differ from the Parquet |
| `E_FREQ_MIX` | Mixed frequencies in one evaluation, including differing q from `x_periods_per_year` |
| `E_PERIODS` | `x_periods_per_year` present but not a positive integer |

**Warnings:**

| Code | Meaning |
|---|---|
| `W_SHARPE_MISMATCH` | \|recomputed − logged `sharpe_annual`\| > 0.01, recomputed with the row's q |
| `W_DIRTY` | Dirty git tree |
| `W_SHORT` | `n_obs` < 60 |
| `W_FAILED` | Failed or manual rows are present |
| `W_SUMMARY_ONLY` | No Parquet file exists, so the series were not checked |

`E_SHARPE` and `W_SUMMARY_ONLY` were added before the freeze; see [decisions](decisions.md).

**Fresh clones.** The Parquet files are git-ignored, so a CI checkout usually has none. A log with no Parquet at all is in summary-only mode: the CSV is fully checked and the log passes with `W_SUMMARY_ONLY`. If at least one series exists, every other series must exist too, or `E_RETURNS_MISSING` fires. `--strict` is the default and exits 1 on any error; `--no-strict` lists the issues and exits 0.

## How N, the variance of trial Sharpes, and the DSR are computed

- **Scope.** Select rows by `--family`, default all families. All rows in scope must share one q, otherwise `E_FREQ_MIX`. That covers rows whose `x_periods_per_year` values differ, and rows with and without the column whose effective q differ, such as 365 next to an inferred 252. The file-level validator applies the rule per family, because families at different frequencies may share a log.
- **Frequency.** If a row has `x_periods_per_year`, q is that value, used for Sharpe annualisation, PSR, DSR, MinTRL and the haircut. Otherwise, for each valid trial, q comes from the median spacing of its dates in calendar days: ≤ 4 → 252 (daily), 5–10 → 52 (weekly), 25–35 → 12 (monthly), anything else → `E_FREQ_MIX`. In summary-only mode the spacing is `(end − start)/(n_obs − 1)`.
- **Per-trial Sharpe.** For each row with a series, SRᵢ = mean(retᵢ)/std(retᵢ, ddof=1), per period, on that trial's own window. If no Parquet files are available at all (summary-only mode), SRᵢ = `sharpe_annual`ᵢ/√q and the report states `variance_source = "summary"`.
- **N.** `raw`: number of rows in scope, including failed and manual rows. `dedup`: unique dedup keys. `clustered`: effective N on the aligned matrix (needs at least 252 common observations); failed and manual rows are added to the cluster count as singletons.
- **V.** V̂ = Σ(SRᵢ − mean(SR))²/(M − 1) over the M valid trials, in per-period units, and reported also as V̂·q (annual). Valid means `n_obs > 0`. If M < 2, V̂ falls back to the null variance of the SR estimator 1/(T̄ − 1), with T̄ the mean `n_obs`, and the report flags `variance_fallback = true`.
- **Selected trial.** `--selected run_id`; else the row with `x_selected = true`; else the row with the highest `sharpe_annual`.
- **SR₀ and DSR.** SR₀ = √V̂·((1 − γ)Φ⁻¹(1 − 1/N) + γΦ⁻¹(1 − 1/(N·e))). DSR = PSR(SR₀), using the selected trial's per-period SR, its `n_obs`, skew and non-excess kurtosis computed from its Parquet series. Summary-only mode assumes skew 0 and kurtosis 3, and flags it.
- **PBO and snooping tests.** Built from `returns_matrix()`, an inner join of all valid series on date. S = 16 needs at least 16·20 = 320 common observations, otherwise PBO is "not computed" and the verdict can be at most INCONCLUSIVE. Snooping tests use a zero benchmark unless the family provides `x_benchmark_run_id`.
- **Haircut.** Harvey–Liu on the selected trial's annual SR with `n_tests` = N. A sample of under two months reports "not computed".

## Worked example

`tests/fixtures/trial_log_example/` holds five rows in two families. The scope below is family `ma_demo`: three trials with eight daily observations each, and one aborted trial. The numbers come from `tests/fixtures/trial_log_hand_calc.py`, which uses only the Python standard library and shares no code with `overfitlab`. `tests/unit/test_triallog.py::test_worked_example` asserts them to 1e-9.

| Quantity | Calculation | Value |
|---|---|---|
| N | 4 rows, including the aborted one | 4 |
| M | Valid trials | 3 |
| V̂ | Sample variance of the three per-period Sharpe ratios | 0.414159210388 |
| SR₀ | √V̂·((1 − γ)Φ⁻¹(3/4) + γΦ⁻¹(1 − 1/(4e))) | 0.677095775287 |
| SR | Selected trial, the highest `sharpe_annual` | 0.784662123195 |
| γ₃ | Skewness of the selected trial | −0.346308523254 |
| γ₄ | Non-excess kurtosis of the selected trial | 2.018844051285 |
| z | (SR − SR₀)·√7 / √(1 − γ₃·SR + (γ₄ − 1)/4·SR²) | 0.238109264691 |
| DSR | Φ(z) | 0.594101825407 |

Naively, the selected trial has a per-period Sharpe ratio of 0.78. After three trials with widely spread results and one failure, the bar is 0.68, so the deflated confidence is only 0.59.

## Reference writer

Copy this into a repo that does not depend on `overfitlab`. It needs only `pyarrow`, and `tests/unit/test_triallog.py` runs this exact block against the validator.

<!-- reference-writer -->
```python
"""Trial-log contract v1.0 reference writer. Needs pyarrow."""

import csv
import json
import math
import statistics
import subprocess
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

COLUMNS = (
    "run_id",
    "timestamp",
    "git_sha",
    "strategy_family",
    "params_json",
    "start",
    "end",
    "n_obs",
    "sharpe_annual",
    "returns_path",
    "x_periods_per_year",
    "x_note",
)


def git_sha(log_path: Path) -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    try:
        sha = git("rev-parse", "HEAD").strip()
        root = Path(git("rev-parse", "--show-toplevel").strip())
        log = log_path.resolve().relative_to(root.resolve()).as_posix()
        status = git("status", "--porcelain", "--untracked-files=no", "--", ".", f":(exclude){log}")
    except (OSError, ValueError, subprocess.CalledProcessError):
        return "0" * 40 + "-dirty"
    return f"{sha}-dirty" if status.strip() else sha


def append_row(log_path: Path, row: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, restval="", lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def base_row(log_path: Path, family: str, params: dict[str, object]) -> dict[str, str]:
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(log_path),
        "strategy_family": family,
        "params_json": json.dumps(
            params, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ),
    }


def log_trial(
    log_path: Path,
    family: str,
    params: dict[str, object],
    dates: list[date],
    returns: list[float],
    periods_per_year: int = 252,
) -> str:
    """Write the series, then append the row. Dates must be strictly increasing."""
    row = base_row(log_path, family, params)
    relative = f"runs/{row['run_id']}/returns.parquet"
    target = log_path.parent / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"date": pa.array(dates, pa.date32()), "ret": pa.array(returns, pa.float64())})
    pq.write_table(table, target)
    std = statistics.stdev(returns)
    sharpe = statistics.fmean(returns) / std * math.sqrt(periods_per_year) if std > 0 else 0.0
    row |= {
        "start": dates[0].isoformat(),
        "end": dates[-1].isoformat(),
        "n_obs": str(len(returns)),
        "sharpe_annual": f"{sharpe:.10g}",
        "returns_path": relative,
        "x_periods_per_year": str(periods_per_year),
    }
    append_row(log_path, row)
    return row["run_id"]


def log_failed(log_path: Path, family: str, params: dict[str, object], note: str) -> str:
    """A failed or aborted run still counts in N."""
    row = base_row(log_path, family, params) | {"n_obs": "0", "x_note": note}
    append_row(log_path, row)
    return row["run_id"]
```
<!-- /reference-writer -->
