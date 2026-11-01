"""Cross-project trial-log contract v1.0: writer, reader, validator and evaluator.

The contract is published in ``docs/trial-log.md`` and described for machines in
``trial_log_schema.json``. It is frozen: later changes only add optional ``x_`` columns.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from overfitlab._arrays import as_vector
from overfitlab.deflated import deflated_sharpe_ratio
from overfitlab.haircut import haircut_sharpe
from overfitlab.pbo import PBOResult, pbo_cscv
from overfitlab.sharpe import annualize_sharpe, return_moments, sharpe_ratio
from overfitlab.snooping import SPAResult, StepMResult
from overfitlab.verdict import MIN_OBS_PER_BLOCK, VerdictConfig, decide, judge, run_snooping

if TYPE_CHECKING:
    import pyarrow as pa

logger = logging.getLogger(__name__)

F64 = NDArray[np.float64]

TRIAL_LOG_COLUMNS: tuple[str, ...] = (
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
)
TRIAL_LOG_VERSION: str = "1.0"

_DEFAULT_EXTRA = ("x_periods_per_year", "x_note")
_RUN_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}(-dirty)?$")
_FAMILY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NO_GIT_SHA = "0" * 40 + "-dirty"
_SHARPE_TOLERANCE = 0.01
_SHORT_OBS = 60
_MIN_CLUSTER_OBS = 252
_MIN_SILHOUETTE = 0.05


class TrialLogError(ValueError):
    """A trial log failed validation or cannot be evaluated.

    Attributes
    ----------
    issues : list of TrialLogIssue
        The error-level issues.
    """

    def __init__(self, message: str, issues: Sequence[TrialLogIssue] = ()) -> None:
        super().__init__(message)
        self.issues = list(issues)


def canonical_params_json(params: Mapping[str, object]) -> str:
    """Return the canonical JSON text of a parameter mapping.

    Parameters
    ----------
    params : mapping
        Choices that define the trial. JSON types only.

    Returns
    -------
    str
        Text with sorted keys, no spaces, ASCII only and no NaN.

    Raises
    ------
    ValueError
        If a value is NaN or infinite.
    TypeError
        If a value is not a JSON type.

    Examples
    --------
    >>> canonical_params_json({"b": 1, "a": "x"})
    '{"a":"x","b":1}'
    """
    return json.dumps(
        params, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _git_sha(repo_dir: Path, exclude: Sequence[Path]) -> str:
    """Return HEAD, marked dirty if the tree has changes other than the log itself."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        excludes = [f":(exclude){path.as_posix()}" for path in exclude]
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", ".", *excludes],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return _NO_GIT_SHA
    return f"{sha}-dirty" if status else sha


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_day_array(dates: ArrayLike) -> NDArray[np.datetime64]:
    if hasattr(dates, "to_numpy"):
        dates = dates.to_numpy()
    return np.asarray(dates, dtype="datetime64[D]")


def _write_parquet(path: Path, dates: NDArray[np.datetime64], returns: F64) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"date": pa.array(dates, type=pa.date32()), "ret": pa.array(returns)})
    pq.write_table(table, path)


class TrialLogger:
    """Append trials to a contract v1.0 log.

    Call :meth:`log` inside the shared evaluation function, so that no evaluation can
    skip it. New logs get the columns ``x_periods_per_year`` and ``x_note`` plus any
    ``extra`` keys of the first row. An existing log keeps its header, because the file
    is append-only.

    Parameters
    ----------
    log_path : str or pathlib.Path, default "results/trial_log.csv"
        Log file.
    returns_dir : str or pathlib.Path, default "results/runs"
        Folder of the Parquet return series. Must be ``runs`` next to the log.
    repo_dir : str or pathlib.Path, default "."
        Git repository whose commit is recorded.

    Raises
    ------
    ValueError
        If ``returns_dir`` is not the folder ``runs`` next to ``log_path``.

    Examples
    --------
    >>> import tempfile
    >>> folder = Path(tempfile.mkdtemp())
    >>> trials = TrialLogger(folder / "trial_log.csv", folder / "runs", repo_dir=folder)
    >>> run_id = trials.log(strategy_family="demo", params={"n": 5},
    ...                     dates=["2024-01-02", "2024-01-03", "2024-01-04"],
    ...                     returns=[0.01, -0.02, 0.03])
    >>> len(run_id)
    36
    """

    def __init__(
        self,
        log_path: str | Path = "results/trial_log.csv",
        returns_dir: str | Path = "results/runs",
        *,
        repo_dir: str | Path = ".",
    ) -> None:
        self.log_path = Path(log_path)
        self.returns_dir = Path(returns_dir)
        self.repo_dir = Path(repo_dir)
        if self.returns_dir.resolve() != self.log_path.resolve().parent / "runs":
            raise ValueError("returns_dir must be the folder 'runs' next to log_path")

    def _sha(self) -> str:
        repo = self.repo_dir.resolve()
        log = self.log_path.resolve()
        exclude = [log.relative_to(repo)] if log.is_relative_to(repo) else []
        return _git_sha(repo, exclude)

    def _header(self, extra_keys: Sequence[str]) -> list[str]:
        for key in extra_keys:
            if not key.startswith("x_"):
                raise ValueError(f"extra column {key!r} must start with x_")
        if self.log_path.exists() and self.log_path.stat().st_size > 0:
            with self.log_path.open(encoding="utf-8", newline="") as fh:
                header = next(csv.reader(fh))
            unknown = [key for key in extra_keys if key not in header]
            if unknown:
                raise ValueError(f"log has no columns {unknown}; its header is append-only")
            return header
        return [*TRIAL_LOG_COLUMNS, *_DEFAULT_EXTRA, *sorted(set(extra_keys) - set(_DEFAULT_EXTRA))]

    def _append(self, values: Mapping[str, str]) -> None:
        extra_keys = [key for key in values if key not in TRIAL_LOG_COLUMNS]
        header = self._header(extra_keys)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.log_path.exists() or self.log_path.stat().st_size == 0
        with self.log_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, restval="", lineterminator="\n")
            if is_new:
                writer.writeheader()
            writer.writerow(values)

    def log(
        self,
        *,
        strategy_family: str,
        params: Mapping[str, object],
        dates: ArrayLike,
        returns: ArrayLike,
        periods_per_year: int = 252,
        extra: Mapping[str, str] | None = None,
    ) -> str:
        """Write the return series and append one row.

        Parameters
        ----------
        strategy_family : str
            Group of related trials.
        params : mapping
            All choices that define the trial.
        dates : array_like
            Return dates, strictly increasing.
        returns : array_like
            Net simple returns per period, in decimal.
        periods_per_year : int, default 252
            Periods per year, written as ``x_periods_per_year``.
        extra : mapping, optional
            More ``x_`` columns. They must exist in the header of an existing log.

        Returns
        -------
        str
            The ``run_id``.

        Raises
        ------
        ValueError
            If an argument breaks the contract.
        """
        if not _FAMILY.match(strategy_family):
            raise ValueError(f"invalid strategy_family {strategy_family!r}")
        if periods_per_year < 1:
            raise ValueError("periods_per_year must be a positive integer")
        days = _to_day_array(dates)
        rets = as_vector(returns, name="returns")
        if days.size != rets.size:
            raise ValueError("dates and returns differ in length")
        if rets.size < 2:
            raise ValueError("at least two observations are needed; use log_failed otherwise")
        if np.isnat(days).any() or not (np.diff(days) > np.timedelta64(0, "D")).all():
            raise ValueError("dates must be strictly increasing, with no nulls or duplicates")

        run_id = str(uuid.uuid4())
        _write_parquet(self.returns_dir / run_id / "returns.parquet", days, rets)
        sharpe = annualize_sharpe(float(sharpe_ratio(rets)), periods_per_year)
        self._append(
            {
                "run_id": run_id,
                "timestamp": _utc_now(),
                "git_sha": self._sha(),
                "strategy_family": strategy_family,
                "params_json": canonical_params_json(params),
                "start": str(days[0]),
                "end": str(days[-1]),
                "n_obs": str(rets.size),
                "sharpe_annual": f"{sharpe:.10g}",
                "returns_path": f"runs/{run_id}/returns.parquet",
                "x_periods_per_year": str(periods_per_year),
                **(extra or {}),
            }
        )
        return run_id

    def log_failed(self, *, strategy_family: str, params: Mapping[str, object], note: str) -> str:
        """Append a failed or aborted trial, which still counts in N.

        Parameters
        ----------
        strategy_family : str
            Group of related trials.
        params : mapping
            All choices that define the trial.
        note : str
            Why it failed, written as ``x_note``.

        Returns
        -------
        str
            The ``run_id``.

        Raises
        ------
        ValueError
            If ``strategy_family`` breaks the contract.
        """
        if not _FAMILY.match(strategy_family):
            raise ValueError(f"invalid strategy_family {strategy_family!r}")
        run_id = str(uuid.uuid4())
        self._append(
            {
                "run_id": run_id,
                "timestamp": _utc_now(),
                "git_sha": self._sha(),
                "strategy_family": strategy_family,
                "params_json": canonical_params_json(params),
                "n_obs": "0",
                "x_note": note,
            }
        )
        return run_id


def add_manual_trial(
    log_path: str | Path,
    *,
    strategy_family: str,
    params: Mapping[str, object],
    note: str,
    start: str = "",
    end: str = "",
    sharpe_annual: float | None = None,
    repo_dir: str | Path = ".",
) -> str:
    """Append a hand iteration that left no saved return series.

    The row counts in N and is excluded from the variance of Sharpe ratios.

    Parameters
    ----------
    log_path : str or pathlib.Path
        Log file.
    strategy_family : str
        Group of related trials.
    params : mapping
        All choices that define the trial.
    note : str
        What was tried, written as ``x_note``.
    start, end : str, default ""
        First and last dates, ``YYYY-MM-DD``, if known.
    sharpe_annual : float, optional
        Sharpe ratio seen at the time, if known.
    repo_dir : str or pathlib.Path, default "."
        Git repository whose commit is recorded.

    Returns
    -------
    str
        The ``run_id``.

    Raises
    ------
    ValueError
        If an argument breaks the contract.

    Examples
    --------
    >>> import tempfile
    >>> log = Path(tempfile.mkdtemp()) / "trial_log.csv"
    >>> run_id = add_manual_trial(log, strategy_family="demo", params={"idea": 1},
    ...                           note="tried by hand", repo_dir=log.parent)
    >>> read_trial_log(log).n_trials()
    1.0
    """
    if not _FAMILY.match(strategy_family):
        raise ValueError(f"invalid strategy_family {strategy_family!r}")
    for value in (start, end):
        if value and _parse_date(value) is None:
            raise ValueError(f"invalid date {value!r}")
    path = Path(log_path)
    logger_ = TrialLogger(path, path.parent / "runs", repo_dir=repo_dir)
    run_id = str(uuid.uuid4())
    logger_._append(
        {
            "run_id": run_id,
            "timestamp": _utc_now(),
            "git_sha": logger_._sha(),
            "strategy_family": strategy_family,
            "params_json": canonical_params_json(params),
            "start": start,
            "end": end,
            "n_obs": "0",
            "sharpe_annual": "" if sharpe_annual is None else f"{sharpe_annual:.10g}",
            "x_note": note,
        }
    )
    return run_id


@dataclass(frozen=True)
class TrialLogIssue:
    """One finding of the validator.

    Attributes
    ----------
    code : str
        ``E_*`` for errors and ``W_*`` for warnings.
    severity : {"error", "warning"}
        Severity.
    row : int or None
        One-based data row, or None for the whole file.
    message : str
        Explanation.
    """

    code: str
    severity: Literal["error", "warning"]
    row: int | None
    message: str


@dataclass(frozen=True)
class TrialSharpes:
    """Per-period Sharpe ratios of the valid trials in a scope.

    Attributes
    ----------
    run_ids : tuple of str
        Trials, in log order.
    sr : numpy.ndarray
        Per-period Sharpe ratios.
    periods_per_year : int
        Periods per year shared by the scope.
    source : {"returns", "summary"}
        Whether they come from the Parquet series or from ``sharpe_annual``.
    """

    run_ids: tuple[str, ...]
    sr: F64
    periods_per_year: int
    source: Literal["returns", "summary"]


def _error(code: str, row: int | None, message: str) -> TrialLogIssue:
    return TrialLogIssue(code, "error", row, message)


def _warning(code: str, row: int | None, message: str) -> TrialLogIssue:
    return TrialLogIssue(code, "warning", row, message)


def _parse_date(value: str) -> date | None:
    if not _DATE.match(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    return int(value) if re.fullmatch(r"\d+", value) else None


def _parse_float(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], list[TrialLogIssue]]:
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            table = list(csv.reader(fh))
    except UnicodeDecodeError:
        return [], [], [_error("E_COLUMNS", None, "file is not UTF-8")]
    if not table:
        return [], [], [_error("E_COLUMNS", None, "file is empty")]
    header = table[0]
    issues: list[TrialLogIssue] = []
    if tuple(header[: len(TRIAL_LOG_COLUMNS)]) != TRIAL_LOG_COLUMNS:
        issues.append(_error("E_COLUMNS", None, f"required columns differ: {header}"))
    extra = [name for name in header[len(TRIAL_LOG_COLUMNS) :] if not name.startswith("x_")]
    if extra:
        issues.append(_error("E_COLUMNS", None, f"unknown non-x_ columns: {extra}"))
    rows: list[dict[str, str]] = []
    for number, cells in enumerate(table[1:], start=1):
        if len(cells) != len(header):
            issues.append(_error("E_COLUMNS", number, f"expected {len(header)} cells"))
            continue
        rows.append(dict(zip(header, cells, strict=True)))
    return header, rows, issues


def _check_row(number: int, row: Mapping[str, str], seen: set[str]) -> list[TrialLogIssue]:
    """Check the fields that need no return series."""
    issues: list[TrialLogIssue] = []
    if not _RUN_ID.match(row["run_id"]):
        issues.append(_error("E_RUN_ID", number, f"not a lower-case UUID v4: {row['run_id']!r}"))
    elif row["run_id"] in seen:
        issues.append(_error("E_RUN_ID", number, f"duplicate run_id {row['run_id']}"))
    seen.add(row["run_id"])

    stamp = row["timestamp"]
    if not _TIMESTAMP.match(stamp) or _is_bad_timestamp(stamp):
        issues.append(_error("E_TIMESTAMP", number, f"invalid timestamp {stamp!r}"))
    if not _GIT_SHA.match(row["git_sha"]):
        issues.append(_error("E_GIT_SHA", number, f"invalid git_sha {row['git_sha']!r}"))
    if not _FAMILY.match(row["strategy_family"]):
        issues.append(_error("E_FAMILY", number, f"invalid family {row['strategy_family']!r}"))
    issues += _check_params(number, row["params_json"])

    n_obs = _parse_int(row["n_obs"])
    if n_obs is None:
        issues.append(_error("E_NOBS", number, f"invalid n_obs {row['n_obs']!r}"))
    issues += _check_dates(number, row, n_obs)
    issues += _check_summary(number, row, n_obs)

    periods = row.get("x_periods_per_year", "")
    if periods and (_parse_int(periods) or 0) < 1:
        issues.append(_error("E_PERIODS", number, f"not a positive integer: {periods!r}"))
    return issues


def _is_bad_timestamp(stamp: str) -> bool:
    try:
        datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return True
    return False


def _check_params(number: int, text: str) -> list[TrialLogIssue]:
    try:
        params = json.loads(text)
        canonical = isinstance(params, dict) and canonical_params_json(params) == text
    except ValueError:
        canonical = False
    if canonical:
        return []
    return [_error("E_PARAMS_JSON", number, "params_json is not canonical JSON")]


def _check_dates(number: int, row: Mapping[str, str], n_obs: int | None) -> list[TrialLogIssue]:
    start, end = row["start"], row["end"]
    if not start and not end and not n_obs:
        return []
    first, last = _parse_date(start), _parse_date(end)
    if first is None or last is None:
        return [_error("E_DATES", number, f"invalid dates {start!r}, {end!r}")]
    if first > last:
        return [_error("E_DATES", number, f"start {start} is after end {end}")]
    return []


def _check_summary(number: int, row: Mapping[str, str], n_obs: int | None) -> list[TrialLogIssue]:
    issues: list[TrialLogIssue] = []
    sharpe, path = row["sharpe_annual"], row["returns_path"]
    if sharpe and _parse_float(sharpe) is None or (n_obs and not sharpe):
        issues.append(_error("E_SHARPE", number, f"invalid sharpe_annual {sharpe!r}"))
    expected = f"runs/{row['run_id']}/returns.parquet"
    if (n_obs and path != expected) or (not n_obs and path):
        issues.append(_error("E_RETURNS_MISSING", number, f"expected returns_path {expected}"))
    return issues


@dataclass(frozen=True)
class _Series:
    dates: NDArray[np.datetime64]
    ret: F64


def _read_table(path: Path) -> pa.Table:
    import pyarrow.parquet as pq

    return pq.read_table(path)


def _dates_as_days(column: pa.ChunkedArray) -> NDArray[np.datetime64] | None:
    import pyarrow as pa

    if pa.types.is_date32(column.type):
        return cast(NDArray[np.datetime64], column.to_numpy().astype("datetime64[D]"))
    if pa.types.is_timestamp(column.type):
        stamps = column.to_numpy().astype("datetime64[ns]")
        days = stamps.astype("datetime64[D]")
        if (stamps[~np.isnat(stamps)] == days[~np.isnat(days)].astype("datetime64[ns]")).all():
            return cast(NDArray[np.datetime64], days)
    return None


def _load_series(path: Path) -> tuple[_Series | None, tuple[str, str] | None]:
    """Read a Parquet series, or return the contract error it breaks."""
    import pyarrow as pa

    table = _read_table(path)
    if "date" not in table.column_names or "ret" not in table.column_names:
        return None, ("E_RETURNS_SCHEMA", f"columns date and ret needed, got {table.column_names}")
    days = _dates_as_days(table["date"])
    if days is None or not pa.types.is_float64(table["ret"].type):
        message = f"bad types: date {table['date'].type}, ret {table['ret'].type}"
        return None, ("E_RETURNS_SCHEMA", message)
    if table["date"].null_count or table["ret"].null_count:
        return None, ("E_RETURNS_VALUES", "null date or ret")
    ret = table["ret"].to_numpy()
    if not np.isfinite(ret).all():
        return None, ("E_RETURNS_VALUES", "NaN or inf in ret")
    if days.size > 1 and not (np.diff(days) > np.timedelta64(0, "D")).all():
        return None, ("E_RETURNS_ORDER", "dates not strictly increasing")
    return _Series(days, ret), None


def _frequency_from_spacing(spacing_days: float) -> int | None:
    if spacing_days <= 4:
        return 252
    if 5 <= spacing_days <= 10:
        return 52
    if 25 <= spacing_days <= 35:
        return 12
    return None


def _infer_periods(dates: NDArray[np.datetime64]) -> int | None:
    """Infer q from the median spacing of the dates, in whole calendar days."""
    if dates.size < 2:
        return None
    gaps = np.sort(np.diff(dates).astype(np.int64))
    return _frequency_from_spacing(float(gaps[(gaps.size - 1) // 2]))


def _infer_periods_summary(row: Mapping[str, str]) -> int | None:
    """Infer q from the window when there is no series."""
    first, last, n_obs = (
        _parse_date(row["start"]),
        _parse_date(row["end"]),
        _parse_int(row["n_obs"]),
    )
    if first is None or last is None or n_obs is None or n_obs < 2:
        return None
    return _frequency_from_spacing((last - first).days / (n_obs - 1))


def _declared_periods(row: Mapping[str, str]) -> int | None:
    value = _parse_int(row.get("x_periods_per_year", ""))
    return value if value else None


def _check_series(
    number: int, row: Mapping[str, str], base_dir: Path
) -> tuple[list[TrialLogIssue], int | None]:
    """Check one row against its Parquet file. Returns the issues and the row's q."""
    path = base_dir / row["returns_path"]
    series, problem = _load_series(path)
    if series is None:
        assert problem is not None
        return [_error(problem[0], number, problem[1])], None
    issues: list[TrialLogIssue] = []
    if _parse_int(row["n_obs"]) != series.ret.size:
        issues.append(_error("E_NOBS", number, f"Parquet has {series.ret.size} rows"))
    dates_ok = not _check_dates(number, row, series.ret.size)
    if (
        dates_ok
        and series.ret.size
        and (str(series.dates[0]) != row["start"] or str(series.dates[-1]) != row["end"])
    ):
        window = f"{series.dates[0]} to {series.dates[-1]}"
        issues.append(_error("E_WINDOW", number, f"Parquet covers {window}"))
    periods = _declared_periods(row) or _infer_periods(series.dates)
    if periods is None:
        issues.append(_error("E_FREQ_MIX", number, "cannot infer the frequency from the dates"))
    logged = _parse_float(row["sharpe_annual"])
    if periods and logged is not None and series.ret.size >= 2:
        recomputed = annualize_sharpe(float(sharpe_ratio(series.ret)), periods)
        if abs(recomputed - logged) > _SHARPE_TOLERANCE:
            message = f"logged {logged}, recomputed {recomputed:.6f}"
            issues.append(_warning("W_SHARPE_MISMATCH", number, message))
    return issues, periods


def _check_row_warnings(number: int, row: Mapping[str, str]) -> list[TrialLogIssue]:
    issues: list[TrialLogIssue] = []
    if row["git_sha"].endswith("-dirty"):
        issues.append(_warning("W_DIRTY", number, "working tree was dirty"))
    n_obs = _parse_int(row["n_obs"])
    if n_obs == 0:
        issues.append(_warning("W_FAILED", number, "failed or manual row: no return series"))
    elif n_obs is not None and n_obs < _SHORT_OBS:
        issues.append(_warning("W_SHORT", number, f"only {n_obs} observations"))
    return issues


def _check_family_frequency(
    periods: Mapping[int, int], rows: Sequence[Mapping[str, str]]
) -> list[TrialLogIssue]:
    by_family: dict[str, set[int]] = {}
    for number, value in periods.items():
        by_family.setdefault(rows[number - 1]["strategy_family"], set()).add(value)
    return [
        _error("E_FREQ_MIX", None, f"family {family} mixes periods per year {sorted(values)}")
        for family, values in by_family.items()
        if len(values) > 1
    ]


def validate_trial_log(path: str | Path) -> list[TrialLogIssue]:
    r"""Check a trial log against contract v1.0.

    A log with no Parquet file at all, such as a fresh clone, is in summary-only mode:
    the CSV is checked and ``W_SUMMARY_ONLY`` says the series were not. If some series
    exist, a missing one is the error ``E_RETURNS_MISSING``.

    Parameters
    ----------
    path : str or pathlib.Path
        Log file.

    Returns
    -------
    list of TrialLogIssue
        Errors and warnings, empty for a clean log.

    Examples
    --------
    >>> import tempfile
    >>> log = Path(tempfile.mkdtemp()) / "trial_log.csv"
    >>> _ = log.write_text("run_id\n", encoding="utf-8")
    >>> [issue.code for issue in validate_trial_log(log)]
    ['E_COLUMNS']
    """
    log_path = Path(path)
    _, rows, issues = _read_csv(log_path)
    if any(issue.row is None for issue in issues) or not rows and issues:
        return issues

    seen: set[str] = set()
    for number, row in enumerate(rows, start=1):
        issues += _check_row(number, row, seen)

    base_dir = log_path.parent
    with_series = [
        (number, row)
        for number, row in enumerate(rows, start=1)
        if _parse_int(row["n_obs"]) and row["returns_path"]
    ]
    present = [(n, r) for n, r in with_series if (base_dir / r["returns_path"]).exists()]
    periods: dict[int, int] = {}
    if with_series and not present:
        issues.append(_warning("W_SUMMARY_ONLY", None, "no Parquet files; series not checked"))
        for number, row in with_series:
            inferred = _declared_periods(row) or _infer_periods_summary(row)
            if inferred is None:
                issues.append(_error("E_FREQ_MIX", number, "cannot infer the frequency"))
            else:
                periods[number] = inferred
    else:
        for number, row in with_series:
            if (base_dir / row["returns_path"]).exists():
                row_issues, value = _check_series(number, row, base_dir)
                issues += row_issues
                if value is not None:
                    periods[number] = value
            else:
                issues.append(_error("E_RETURNS_MISSING", number, row["returns_path"]))
    issues += _check_family_frequency(periods, rows)
    for number, row in enumerate(rows, start=1):
        issues += _check_row_warnings(number, row)
    return issues


def _mean_silhouette(distances: F64, labels: NDArray[np.int64]) -> float:
    """Mean silhouette for a precomputed distance matrix; singletons score zero."""
    n_clusters = int(labels.max())
    members = np.zeros((labels.size, n_clusters))
    members[np.arange(labels.size), labels - 1] = 1.0
    sizes = members.sum(axis=0)
    totals = distances @ members
    rows = np.arange(labels.size)
    own = labels - 1
    within = totals[rows, own] / np.maximum(sizes[own] - 1.0, 1.0)
    between = totals / np.where(sizes > 0, sizes, np.inf)
    between[rows, own] = np.inf
    nearest = between.min(axis=1)
    scores = (nearest - within) / np.maximum(np.maximum(nearest, within), 1e-12)
    scores[sizes[own] == 1.0] = 0.0
    return float(scores.mean())


def _effective_trials(returns: F64) -> int:
    """Estimate independent trials by clustering return correlations.

    Follows the idea of Lopez de Prado and Lewis (2019): hierarchical clustering on the
    distance ``sqrt((1 - rho) / 2)``, with the number of clusters chosen by the largest mean
    silhouette. With no structure to find, every trial counts as independent.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n_trials = returns.shape[1]
    if n_trials < 3:
        return int(n_trials)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.nan_to_num(np.corrcoef(returns, rowvar=False))
    distances = np.sqrt(np.clip(0.5 * (1.0 - correlation), 0.0, None))
    np.fill_diagonal(distances, 0.0)
    tree = linkage(squareform(distances, checks=False), method="average")
    candidates = np.unique(np.geomspace(2, n_trials - 1, num=min(n_trials - 2, 40)).astype(int))
    best_count, best_score = int(n_trials), _MIN_SILHOUETTE
    for count in candidates:
        labels = fcluster(tree, int(count), criterion="maxclust").astype(np.int64)
        score = _mean_silhouette(distances, labels)
        if score > best_score:
            best_count, best_score = int(labels.max()), score
    return best_count


class TrialLog:
    """A loaded trial log, or a filtered part of one.

    Parameters
    ----------
    rows : sequence of mapping
        Rows as strings, keyed by column name.
    base_dir : pathlib.Path
        Folder of the CSV, which ``returns_path`` is relative to.

    Attributes
    ----------
    rows : tuple of mapping
        Rows as strings, keyed by column name.
    base_dir : pathlib.Path
        Folder of the CSV.

    Examples
    --------
    >>> log = TrialLog([], Path("."))
    >>> log.n_trials()
    0.0
    """

    def __init__(self, rows: Sequence[Mapping[str, object]], base_dir: Path) -> None:
        self.rows: tuple[Mapping[str, object], ...] = tuple(rows)
        self.base_dir = base_dir
        self._cache: dict[str, _Series] = {}

    def _valid_rows(self) -> list[Mapping[str, object]]:
        return [row for row in self.rows if (_parse_int(str(row["n_obs"])) or 0) > 0]

    def _series(self, row: Mapping[str, object]) -> _Series:
        run_id = str(row["run_id"])
        if run_id not in self._cache:
            series, problem = _load_series(self.base_dir / str(row["returns_path"]))
            if series is None:
                raise TrialLogError(f"{run_id}: {problem}")
            self._cache[run_id] = series
        return self._cache[run_id]

    def _has_series(self, row: Mapping[str, object]) -> bool:
        path = str(row["returns_path"])
        return bool(path) and (self.base_dir / path).exists()

    def filter(
        self, *, families: Sequence[str] | None = None, exclude_failed: bool = False
    ) -> TrialLog:
        """Return the rows of some families, optionally without failed rows.

        Parameters
        ----------
        families : sequence of str, optional
            Families to keep. Defaults to all.
        exclude_failed : bool, default False
            Drop rows with ``n_obs = 0``.

        Returns
        -------
        TrialLog
            The selected rows.
        """
        rows: Sequence[Mapping[str, object]] = self.rows
        if families is not None:
            rows = [row for row in rows if row["strategy_family"] in families]
        if exclude_failed:
            rows = [row for row in rows if (_parse_int(str(row["n_obs"])) or 0) > 0]
        return TrialLog(rows, self.base_dir)

    def periods_per_year(self) -> int:
        """Return the q shared by the valid rows in scope.

        Returns
        -------
        int
            Periods per year.

        Raises
        ------
        TrialLogError
            With ``E_FREQ_MIX`` if the rows differ or q cannot be inferred.
        """
        values: set[int | None] = set()
        for row in self._valid_rows():
            text_row = {key: str(value) for key, value in row.items()}
            declared = _declared_periods(text_row)
            if declared:
                values.add(declared)
            elif self._has_series(row):
                values.add(_infer_periods(self._series(row).dates))
            else:
                values.add(_infer_periods_summary(text_row))
        if len(values) != 1 or None in values:
            issue = _error("E_FREQ_MIX", None, f"periods per year in scope: {values}")
            raise TrialLogError(f"E_FREQ_MIX: {issue.message}", [issue])
        return next(value for value in values if value is not None)

    def trial_sharpes(
        self, *, source: Literal["auto", "returns", "summary"] = "auto"
    ) -> TrialSharpes:
        """Return the per-period Sharpe ratio of every valid trial in scope.

        Parameters
        ----------
        source : {"auto", "returns", "summary"}, default "auto"
            Use the Parquet series, ``sharpe_annual``, or the series when any exist.

        Returns
        -------
        TrialSharpes
            Run ids, Sharpe ratios, q and the source used.

        Raises
        ------
        TrialLogError
            If the rows in scope do not share one q.
        """
        rows = self._valid_rows()
        q = self.periods_per_year()
        use_series = source == "returns" or (
            source == "auto" and any(self._has_series(row) for row in rows)
        )
        if use_series:
            rows = [row for row in rows if self._has_series(row)]
            values = [float(sharpe_ratio(self._series(row).ret)) for row in rows]
        else:
            rows = [row for row in rows if _parse_float(str(row["sharpe_annual"])) is not None]
            values = [float(str(row["sharpe_annual"])) / math.sqrt(q) for row in rows]
        return TrialSharpes(
            run_ids=tuple(str(row["run_id"]) for row in rows),
            sr=np.asarray(values, dtype=np.float64),
            periods_per_year=q,
            source="returns" if use_series else "summary",
        )

    def n_trials(self, *, mode: Literal["raw", "dedup", "clustered"] = "raw") -> float:
        """Return the number of trials in scope.

        Parameters
        ----------
        mode : {"raw", "dedup", "clustered"}, default "raw"
            ``raw`` counts rows, including failed and manual ones. ``dedup`` counts unique
            family, params, window and commit. ``clustered`` estimates effective
            independent trials.

        Returns
        -------
        float
            Number of trials.

        Raises
        ------
        ValueError
            If ``mode`` is unknown.
        """
        if mode == "raw":
            return float(len(self.rows))
        if mode == "dedup":
            keys = {
                (
                    row["strategy_family"],
                    row["params_json"],
                    row["start"],
                    row["end"],
                    row["git_sha"],
                )
                for row in self.rows
            }
            return float(len(keys))
        if mode == "clustered":
            _, matrix, _ = self.returns_matrix()
            if matrix.shape[0] < _MIN_CLUSTER_OBS:
                raise ValueError(f"clustered needs {_MIN_CLUSTER_OBS} common observations")
            without_series = len(self.rows) - matrix.shape[1]
            return float(_effective_trials(matrix) + without_series)
        raise ValueError(f"unknown mode {mode!r}")

    def variance_of_sharpes(
        self, *, source: Literal["auto", "returns", "summary"] = "auto"
    ) -> float:
        """Return the sample variance of the per-period trial Sharpe ratios.

        With fewer than two valid trials it falls back to the null variance of the
        estimator, ``1 / (T - 1)``, with ``T`` the mean ``n_obs``.

        Parameters
        ----------
        source : {"auto", "returns", "summary"}, default "auto"
            Source of the Sharpe ratios.

        Returns
        -------
        float
            Variance, in per-period units.
        """
        sharpes = self.trial_sharpes(source=source)
        if sharpes.sr.size >= 2:
            return float(np.var(sharpes.sr, ddof=1))
        counts = [int(str(row["n_obs"])) for row in self._valid_rows()]
        return 1.0 / (float(np.mean(counts)) - 1.0)

    def returns_matrix(self) -> tuple[NDArray[np.datetime64], F64, tuple[str, ...]]:
        """Return the aligned return matrix: an inner join of all series on date.

        Returns
        -------
        dates : numpy.ndarray
            Common dates.
        matrix : numpy.ndarray
            ``T x N`` returns.
        run_ids : tuple of str
            Trial of each column.
        """
        rows = [row for row in self._valid_rows() if self._has_series(row)]
        if not rows:
            return np.array([], dtype="datetime64[D]"), np.zeros((0, 0)), ()
        series = [self._series(row) for row in rows]
        common = series[0].dates
        for item in series[1:]:
            common = np.intersect1d(common, item.dates, assume_unique=True)
        columns = [item.ret[np.searchsorted(item.dates, common)] for item in series]
        return common, np.column_stack(columns), tuple(str(row["run_id"]) for row in rows)

    def selected_index(self, run_id: str | None = None) -> int:
        """Return the position in ``rows`` of the selected trial.

        The order is ``run_id``, then a row with ``x_selected = true``, then the highest
        ``sharpe_annual``. Only rows with a track record can be selected by default.

        Parameters
        ----------
        run_id : str, optional
            Trial to select.

        Returns
        -------
        int
            Index into ``rows``.

        Raises
        ------
        ValueError
            If ``run_id`` is not in scope or no row can be selected.
        """
        if run_id is not None:
            for index, row in enumerate(self.rows):
                if row["run_id"] == run_id:
                    return index
            raise ValueError(f"run_id {run_id} is not in scope")
        candidates = [
            (index, row)
            for index, row in enumerate(self.rows)
            if (_parse_int(str(row["n_obs"])) or 0) > 0
            and _parse_float(str(row["sharpe_annual"])) is not None
        ]
        if not candidates:
            raise ValueError("no row has a Sharpe ratio to select")
        marked = [
            pair for pair in candidates if str(pair[1].get("x_selected", "")).lower() == "true"
        ]
        pool = marked or candidates
        return max(pool, key=lambda pair: float(str(pair[1]["sharpe_annual"])))[0]


def read_trial_log(path: str | Path, *, validate: bool = True, strict: bool = True) -> TrialLog:
    """Read a trial log.

    Parameters
    ----------
    path : str or pathlib.Path
        Log file.
    validate : bool, default True
        Run :func:`validate_trial_log` first.
    strict : bool, default True
        Raise on error-level issues. Otherwise they are logged and reading goes on.

    Returns
    -------
    TrialLog
        All rows.

    Raises
    ------
    TrialLogError
        If ``validate`` and ``strict`` are set and the log has errors.

    Examples
    --------
    >>> import tempfile
    >>> folder = Path(tempfile.mkdtemp())
    >>> trials = TrialLogger(folder / "trial_log.csv", folder / "runs", repo_dir=folder)
    >>> _ = trials.log_failed(strategy_family="demo", params={}, note="crashed")
    >>> read_trial_log(folder / "trial_log.csv").n_trials()
    1.0
    """
    log_path = Path(path)
    if validate:
        errors = [i for i in validate_trial_log(log_path) if i.severity == "error"]
        if errors and strict:
            codes = sorted({issue.code for issue in errors})
            raise TrialLogError(f"{log_path}: {', '.join(codes)}", errors)
        for issue in errors:
            logger.warning("%s row %s: %s", issue.code, issue.row, issue.message)
    _, rows, _ = _read_csv(log_path)
    return TrialLog(rows, log_path.parent)


@dataclass(frozen=True)
class TrialLogReport:
    """Evaluation of one scope of a trial log.

    Attributes
    ----------
    families : tuple of str
        Families in scope.
    n_trials : float
        N used in the DSR.
    n_trials_mode : str
        How N was counted.
    n_valid : int
        Trials with a track record, M.
    variance : float
        Variance of per-period trial Sharpe ratios.
    variance_annual : float
        ``variance * q``.
    variance_source : str
        ``"returns"`` or ``"summary"``.
    variance_fallback : bool
        True when fewer than two valid trials forced the null variance.
    periods_per_year : int
        q of the scope.
    selected_run_id : str
        Selected trial.
    sr_annual : float
        Naive annual Sharpe ratio of the selected trial.
    sr : float
        Its per-period Sharpe ratio.
    sr0 : float
        Per-period threshold.
    sr0_annual : float
        Annual threshold.
    dsr : float
        Deflated Sharpe Ratio.
    z : float
        Score of the DSR.
    n_obs : int
        Observations of the selected trial.
    skew : float
        Its skewness.
    kurtosis : float
        Its non-excess kurtosis.
    moments_assumed : bool
        True in summary-only mode, where skew 0 and kurtosis 3 are assumed.
    haircut_sr_annual : float or None
        Harvey-Liu haircut Sharpe ratio, if the sample spans two months.
    haircut_pct : float or None
        Haircut in percent.
    pbo : float or None
        Probability of Backtest Overfitting, if computed.
    spa_pvalue : float or None
        Consistent SPA p-value, if computed.
    stepm_rejects_selected : bool or None
        Whether the stepdown rejects the selected trial, if computed.
    verdict : str
        ``PASS``, ``FAIL`` or ``INCONCLUSIVE``.
    reasons : mapping
        Outcome per criterion.
    """

    families: tuple[str, ...]
    n_trials: float
    n_trials_mode: str
    n_valid: int
    variance: float
    variance_annual: float
    variance_source: str
    variance_fallback: bool
    periods_per_year: int
    selected_run_id: str
    sr_annual: float
    sr: float
    sr0: float
    sr0_annual: float
    dsr: float
    z: float
    n_obs: int
    skew: float
    kurtosis: float
    moments_assumed: bool
    haircut_sr_annual: float | None
    haircut_pct: float | None
    pbo: float | None
    spa_pvalue: float | None
    stepm_rejects_selected: bool | None
    verdict: str
    reasons: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the fields as a JSON-ready dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Return the report as one JSON object."""
        return json.dumps(self.to_dict(), sort_keys=True)


def read_wide_returns(path: str | Path) -> tuple[tuple[str, ...], F64]:
    r"""Read a wide table of returns: a ``date`` column and one column per trial.

    Parameters
    ----------
    path : str or pathlib.Path
        Parquet or CSV file.

    Returns
    -------
    names : tuple of str
        Trial columns.
    matrix : numpy.ndarray
        ``T x N`` returns.

    Raises
    ------
    ValueError
        If there is no trial column or a value is not a number.

    Examples
    --------
    >>> import tempfile
    >>> file = Path(tempfile.mkdtemp()) / "wide.csv"
    >>> _ = file.write_text("date,a,b\n2024-01-02,0.1,0.2\n", encoding="utf-8")
    >>> read_wide_returns(file)[0]
    ('a', 'b')
    """
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        table = _read_table(source)
        names = tuple(name for name in table.column_names if name != "date")
        columns = [table[name].to_numpy() for name in names]
    else:
        with source.open(encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            body = list(reader)
        keep = [i for i, name in enumerate(header) if name != "date"]
        names = tuple(header[i] for i in keep)
        columns = [np.array([float(row[i]) for row in body]) for i in keep]
    if not names:
        raise ValueError(f"{source} has no trial columns")
    return names, np.column_stack(columns).astype(np.float64)


def _benchmark_series(log: TrialLog, dates: NDArray[np.datetime64]) -> F64 | None:
    """Return the series named by ``x_benchmark_run_id`` on ``dates``, if a row names one."""
    named = {str(row.get("x_benchmark_run_id", "")) for row in log.rows} - {""}
    if not named:
        return None
    if len(named) > 1:
        raise ValueError(f"more than one x_benchmark_run_id in scope: {sorted(named)}")
    run_id = named.pop()
    path = log.base_dir / "runs" / run_id / "returns.parquet"
    if not path.exists():
        raise ValueError(f"benchmark series {run_id} is missing")
    series, problem = _load_series(path)
    if series is None:
        raise ValueError(f"benchmark series {run_id}: {problem}")
    if not np.isin(dates, series.dates).all():
        raise ValueError(f"benchmark series {run_id} does not cover the trials' dates")
    return series.ret[np.searchsorted(series.dates, dates)]


def _matrix_evidence(
    log: TrialLog, run_id: str, config: VerdictConfig
) -> tuple[PBOResult | None, SPAResult | None, StepMResult | None, int]:
    """Run the tests that need the aligned return matrix, where the log allows them."""
    dates, matrix, run_ids = log.returns_matrix()
    if run_id not in run_ids or matrix.shape[0] < 9:
        return None, None, None, 0
    column = run_ids.index(run_id)
    pbo = None
    if matrix.shape[1] >= 2 and matrix.shape[0] >= config.pbo_partitions * MIN_OBS_PER_BLOCK:
        pbo = pbo_cscv(matrix, n_partitions=config.pbo_partitions)
    benchmark = _benchmark_series(log, dates)
    rel_perf = matrix if benchmark is None else matrix - benchmark[:, np.newaxis]
    spa, stepm = run_snooping(rel_perf, config, 0)
    return pbo, spa, stepm, column


def evaluate_trial_log(
    path: str | Path,
    *,
    families: Sequence[str] | None = None,
    selected: str | None = None,
    n_effective: Literal["raw", "clustered"] = "raw",
    exclude_failed: bool = False,
    config: VerdictConfig = VerdictConfig(),
) -> TrialLogReport:
    """Evaluate a trial log: N, V, SR0, DSR, PBO, snooping tests, haircut and verdict.

    Parameters
    ----------
    path : str or pathlib.Path
        Log file.
    families : sequence of str, optional
        Scope. Defaults to all families, which must share one q.
    selected : str, optional
        ``run_id`` of the selected trial.
    n_effective : {"raw", "clustered"}, default "raw"
        How to count N.
    exclude_failed : bool, default False
        Leave failed and manual rows out of N.
    config : VerdictConfig, optional
        Haircut method and gate thresholds.

    Returns
    -------
    TrialLogReport
        The evaluation of the scope.

    Raises
    ------
    TrialLogError
        If validation fails or the rows in scope do not share one q.
    ValueError
        If no trial in scope has a track record.

    Examples
    --------
    >>> import tempfile
    >>> folder = Path(tempfile.mkdtemp())
    >>> trials = TrialLogger(folder / "trial_log.csv", folder / "runs", repo_dir=folder)
    >>> rng = np.random.default_rng(0)
    >>> days = np.arange("2024-01-01", "2024-07-01", dtype="datetime64[D]")
    >>> for n in (1, 2):
    ...     _ = trials.log(strategy_family="demo", params={"n": n}, dates=days,
    ...                    returns=rng.normal(0.0, 0.01, days.size), periods_per_year=365)
    >>> report = evaluate_trial_log(folder / "trial_log.csv")
    >>> report.n_trials, report.periods_per_year
    (2.0, 365)
    """
    log = read_trial_log(path).filter(families=families, exclude_failed=exclude_failed)
    if not log._valid_rows():
        raise ValueError("no trial in scope has a track record")
    sharpes = log.trial_sharpes()
    q = sharpes.periods_per_year
    n_trials = log.n_trials(mode=n_effective)
    variance = log.variance_of_sharpes()

    row = log.rows[log.selected_index(selected)]
    run_id = str(row["run_id"])
    if run_id not in sharpes.run_ids:
        raise ValueError(f"selected trial {run_id} has no track record")
    sr = float(sharpes.sr[sharpes.run_ids.index(run_id)])
    n_obs = int(str(row["n_obs"]))
    skew, kurtosis, assumed = 0.0, 3.0, sharpes.source == "summary"
    if not assumed:
        series = log._series(row)
        n_obs = series.ret.size
        moments = return_moments(series.ret)
        skew, kurtosis = moments.skew, moments.kurtosis

    deflated = deflated_sharpe_ratio(
        sr, n_obs, n_trials=n_trials, var_trials=variance, skew=skew, kurtosis=kurtosis
    )
    sr_annual = annualize_sharpe(sr, q)
    haircut_sr: float | None = None
    haircut_pct: float | None = None
    try:
        haircut = haircut_sharpe(
            sr_annual,
            n_obs,
            periods_per_year=q,
            n_tests=max(1, round(n_trials)),
            method=config.haircut_method,
        )
        haircut_sr, haircut_pct = haircut.haircut_sr_annual, haircut.haircut_pct
    except ValueError:
        pass

    pbo, spa, stepm, column = _matrix_evidence(log, run_id, replace(config, periods_per_year=q))
    reasons = judge(
        config,
        dsr=deflated,
        pbo=pbo,
        spa=spa,
        stepm=stepm,
        haircut_sr_annual=haircut_sr,
        selected=column,
    )
    return TrialLogReport(
        families=tuple(sorted({str(item["strategy_family"]) for item in log.rows})),
        n_trials=n_trials,
        n_trials_mode=n_effective,
        n_valid=int(sharpes.sr.size),
        variance=variance,
        variance_annual=variance * q,
        variance_source=sharpes.source,
        variance_fallback=sharpes.sr.size < 2,
        periods_per_year=q,
        selected_run_id=run_id,
        sr_annual=sr_annual,
        sr=sr,
        sr0=deflated.sr0,
        sr0_annual=annualize_sharpe(deflated.sr0, q),
        dsr=deflated.dsr,
        z=deflated.z,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurtosis,
        moments_assumed=assumed,
        haircut_sr_annual=haircut_sr,
        haircut_pct=haircut_pct,
        pbo=None if pbo is None else pbo.pbo,
        spa_pvalue=None if spa is None else spa.pvalue_consistent,
        stepm_rejects_selected=None if stepm is None else bool(stepm.rejected[column]),
        verdict=decide(reasons),
        reasons=reasons,
    )
