"""Hand calculation of the worked example, using the standard library only.

This deliberately imports nothing from overfitlab, NumPy or SciPy. It prints the numbers
quoted in docs/trial-log.md and asserted in tests/unit/test_triallog.py::test_worked_example.
Run with ``python tests/fixtures/trial_log_hand_calc.py``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import NormalDist, fmean, pstdev, stdev

FOLDER = Path(__file__).parent / "trial_log_example"
FAMILY = "ma_demo"
GAMMA = 0.5772156649015329


def load_returns(run_id: str) -> list[float]:
    with (FOLDER / "series" / f"{run_id}.csv").open(encoding="utf-8", newline="") as fh:
        return [float(item["ret"]) for item in csv.DictReader(fh)]


def main() -> dict[str, float]:
    with (FOLDER / "trial_log.csv").open(encoding="utf-8", newline="") as fh:
        rows = [row for row in csv.DictReader(fh) if row["strategy_family"] == FAMILY]
    valid = [row for row in rows if int(row["n_obs"]) > 0]
    srs = {row["run_id"]: 0.0 for row in valid}
    for row in valid:
        rets = load_returns(row["run_id"])
        srs[row["run_id"]] = fmean(rets) / stdev(rets)

    n = len(rows)
    m = len(valid)
    mean_sr = fmean(srs.values())
    variance = sum((value - mean_sr) ** 2 for value in srs.values()) / (m - 1)

    normal = NormalDist()
    sr0 = math.sqrt(variance) * (
        (1 - GAMMA) * normal.inv_cdf(1 - 1 / n) + GAMMA * normal.inv_cdf(1 - 1 / (n * math.e))
    )

    selected = max(valid, key=lambda row: float(row["sharpe_annual"]))
    rets = load_returns(selected["run_id"])
    centre = fmean(rets)
    sd = pstdev(rets)
    skew = sum((value - centre) ** 3 for value in rets) / len(rets) / sd**3
    kurtosis = sum((value - centre) ** 4 for value in rets) / len(rets) / sd**4
    sr = srs[selected["run_id"]]
    variance_term = 1 - skew * sr + (kurtosis - 1) / 4 * sr**2
    z = (sr - sr0) * math.sqrt(len(rets) - 1) / math.sqrt(variance_term)
    return {
        "N": float(n),
        "M": float(m),
        "V": variance,
        "SR0": sr0,
        "SR": sr,
        "skew": skew,
        "kurtosis": kurtosis,
        "z": z,
        "DSR": normal.cdf(z),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
