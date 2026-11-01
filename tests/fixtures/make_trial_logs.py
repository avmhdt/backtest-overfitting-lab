"""Write the trial-log fixtures: CSV logs and CSV return series, from a fixed seed.

Parquet files are ignored by git, so tests rebuild them with ``materialise`` in conftest.
Run with ``uv run python tests/fixtures/make_trial_logs.py``.
"""

from __future__ import annotations

import csv
import random
import shutil
import uuid
from datetime import date, timedelta
from pathlib import Path
from statistics import fmean, stdev

ROOT = Path(__file__).parent
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
)
SHA = "0123456789abcdef0123456789abcdef01234567"
STAMP = "2026-10-31T12:00:00Z"


def run_id(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def weekdays(start: date, count: int, step: int = 1) -> list[str]:
    days: list[str] = []
    current = start
    while len(days) < count:
        if step > 1 or current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=step)
    return days


def series(rng: random.Random, count: int, drift: float = 0.0005) -> list[float]:
    return [round(rng.gauss(drift, 0.01), 4) for _ in range(count)]


def sharpe(returns: list[float], periods: int) -> float:
    return fmean(returns) / stdev(returns) * periods**0.5


def row(
    rng: random.Random, family: str, params: str, days: list[str], rets: list[float], periods: int
) -> dict[str, str]:
    rid = run_id(rng)
    return {
        "run_id": rid,
        "timestamp": STAMP,
        "git_sha": SHA,
        "strategy_family": family,
        "params_json": params,
        "start": days[0],
        "end": days[-1],
        "n_obs": str(len(rets)),
        "sharpe_annual": f"{sharpe(rets, periods):.10g}",
        "returns_path": f"runs/{rid}/returns.parquet",
    }


def write_log(
    folder: Path,
    rows: list[dict[str, str]],
    series_by_id: dict[str, tuple[list[str], list[object]]],
    *,
    extra: tuple[str, ...] = (),
    header: tuple[str, ...] | None = None,
    value_column: str = "ret",
) -> None:
    folder.mkdir(parents=True)
    columns = header or (*COLUMNS, *extra)
    with (folder / "trial_log.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=columns, restval="", extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    if series_by_id:
        (folder / "series").mkdir()
    for rid, (days, rets) in series_by_id.items():
        with (folder / "series" / f"{rid}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer_ = csv.writer(fh, lineterminator="\n")
            writer_.writerow(["date", value_column])
            writer_.writerows(zip(days, rets, strict=True))


def worked_example(rng: random.Random) -> None:
    days = weekdays(date(2024, 1, 2), 8)
    rows, data = [], {}
    for family, params, drift in (
        ("ma_demo", '{"fast":5,"slow":60}', 0.008),
        ("ma_demo", '{"fast":10,"slow":60}', 0.004),
        ("ma_demo", '{"fast":20,"slow":60}', 0.0),
        ("rev_demo", '{"lookback":20}', 0.0),
    ):
        rets = series(rng, len(days), drift)
        item = row(rng, family, params, days, rets, 252)
        rows.append(item)
        data[item["run_id"]] = (days, list(rets))
    rows.insert(
        3,
        {
            "run_id": run_id(rng),
            "timestamp": STAMP,
            "git_sha": SHA,
            "strategy_family": "ma_demo",
            "params_json": '{"fast":30,"slow":60}',
            "n_obs": "0",
            "x_note": "aborted: data gap",
        },
    )
    write_log(ROOT / "trial_log_example", rows, data, extra=("x_note",))


def crypto_365(rng: random.Random) -> None:
    days = [(date(2024, 1, 1) + timedelta(days=i)).isoformat() for i in range(120)]
    rows, data = [], {}
    for n in (1, 2, 3):
        rets = series(rng, len(days))
        item = row(rng, "crypto_daily", f'{{"n":{n}}}', days, rets, 365)
        item["x_periods_per_year"] = "365"
        rows.append(item)
        data[item["run_id"]] = (days, list(rets))
    write_log(ROOT / "trial_log_365", rows, data, extra=("x_periods_per_year",))


def bad_logs(rng: random.Random) -> None:
    base = ROOT / "trial_log_bad"
    days = weekdays(date(2024, 1, 2), 70)

    def good(params: str = '{"n":1}', family: str = "demo") -> tuple[dict[str, str], list[object]]:
        rets: list[object] = list(series(rng, len(days)))
        return row(rng, family, params, days, rets, 252), rets  # type: ignore[arg-type]

    def single(code: str, change: dict[str, str]) -> None:
        item, rets = good()
        rid = item["run_id"]
        item.update(change)
        write_log(base / code, [item], {rid: (days, rets)})

    item, rets = good()
    header = list(COLUMNS)
    header[1], header[2] = header[2], header[1]
    write_log(base / "E_COLUMNS", [item], {item["run_id"]: (days, rets)}, header=tuple(header))

    first, rets_a = good()
    second, _ = good('{"n":2}')
    duplicate = first["run_id"]
    second["run_id"] = duplicate
    second["returns_path"] = first["returns_path"]
    write_log(
        base / "E_RUN_ID",
        [first, second],
        {duplicate: (days, rets_a)},
    )

    single("E_TIMESTAMP", {"timestamp": "2026-10-31 12:00:00"})
    single("E_GIT_SHA", {"git_sha": "abc123"})
    single("E_FAMILY", {"strategy_family": "Bad-Family"})
    single("E_PARAMS_JSON", {"params_json": '{"n": 1}'})
    single("E_NOBS", {"n_obs": "69"})
    single("E_SHARPE", {"sharpe_annual": "high"})

    item, rets = good()
    item["start"], item["end"] = item["end"], item["start"]
    write_log(base / "E_DATES", [item], {item["run_id"]: (days, rets)})

    item, rets = good()
    item["start"] = days[1]
    write_log(base / "E_WINDOW", [item], {item["run_id"]: (days, rets)})

    present, rets = good()
    absent, _ = good('{"n":2}')
    write_log(base / "E_RETURNS_MISSING", [present, absent], {present["run_id"]: (days, rets)})

    item, rets = good()
    write_log(
        base / "E_RETURNS_SCHEMA", [item], {item["run_id"]: (days, rets)}, value_column="value"
    )

    item, rets = good()
    write_log(base / "E_RETURNS_VALUES", [item], {item["run_id"]: (days, ["", *rets[1:]])})

    item, rets = good()
    swapped = [days[0], days[2], days[1], *days[3:]]
    write_log(base / "E_RETURNS_ORDER", [item], {item["run_id"]: (swapped, rets)})

    daily, rets_a = good()
    daily["x_periods_per_year"] = "252"
    crypto, rets_b = good('{"n":2}')
    crypto["x_periods_per_year"] = "365"
    crypto["sharpe_annual"] = f"{float(crypto['sharpe_annual']) * (365 / 252) ** 0.5:.10g}"
    write_log(
        base / "E_FREQ_MIX",
        [daily, crypto],
        {daily["run_id"]: (days, rets_a), crypto["run_id"]: (days, rets_b)},
        extra=("x_periods_per_year",),
    )

    item, rets = good()
    item["x_periods_per_year"] = "0"
    write_log(
        base / "E_PERIODS", [item], {item["run_id"]: (days, rets)}, extra=("x_periods_per_year",)
    )


def main() -> None:
    for name in ("trial_log_example", "trial_log_365", "trial_log_bad"):
        shutil.rmtree(ROOT / name, ignore_errors=True)
    rng = random.Random(20261030)
    worked_example(rng)
    crypto_365(rng)
    bad_logs(rng)


if __name__ == "__main__":
    main()
