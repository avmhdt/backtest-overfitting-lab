"""Cross-check against references that really studentise (AC-W-07, owner ruling 2026-11-02).

``bsds.m`` from the MFE Toolbox runs under GNU Octave for SPA and the Reality Check, and R's
``StepwiseTest`` for StepM. Both run in WSL on native Windows, on the same 20 fixed cases as the
``arch`` comparison. This needs local tools, so it is not part of CI. The toolbox has no licence
file, so it is cloned under the data root, pinned to a commit, and never vendored.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import overfitlab as ol
from overfitlab.snooping import stepdown_statistics
from overfitlab_study.validation.arch_xcheck import (
    BLOCK,
    IDENTICAL_TOLERANCE,
    PVALUE_TOLERANCE,
    Case,
    case_data,
    fixed_cases,
)

MFE_REPO = "https://github.com/bashtage/mfe-toolbox.git"
MFE_COMMIT = "9622b6c546bc"
MAX_WORKERS = 8
KINDS = {"lower": "l", "consistent": "c", "upper": "u"}
SCRIPTS = Path("scripts")


def to_wsl(path: Path) -> str:
    """``D:/a/b`` to ``/mnt/d/a/b``. Other paths pass through, for Linux."""
    resolved = path.resolve()
    if not resolved.drive:
        return resolved.as_posix()
    tail = resolved.relative_to(resolved.anchor).as_posix()
    return f"/mnt/{resolved.drive[0].lower()}/{tail}"


def command(arguments: Sequence[str]) -> list[str]:
    """Run through WSL on native Windows."""
    if sys.platform == "win32":
        return ["wsl.exe", "-d", "Ubuntu", "--", *arguments]
    return list(arguments)


def ensure_toolbox(folder: Path) -> Path:
    """Clone the toolbox at the pinned commit and return its bootstrap folder."""
    if not (folder / ".git").exists():
        subprocess.run(["git", "clone", "-q", MFE_REPO, str(folder)], check=True)
    subprocess.run(["git", "-C", str(folder), "checkout", "-q", MFE_COMMIT], check=True)
    return folder / "bootstrap"


def export_case(case: Case, n_boot: int, folder: Path) -> None:
    """Write the case, the resamples and the stepdown statistics as binary files."""
    folder.mkdir(parents=True, exist_ok=True)
    d = case_data(case)
    n_obs, n_strategies = d.shape
    seed = 2000 + case.case_id
    indices = ol.stationary_bootstrap_indices(n_obs, n_boot, mean_block=float(BLOCK), rng=seed)
    tstats, simulated = stepdown_statistics(d, indices, float(BLOCK))
    d.ravel(order="F").astype("<f8").tofile(folder / "d.bin")
    (indices + 1).astype("<i4").tofile(folder / "indices.bin")
    tstats.astype("<f8").tofile(folder / "tstats.bin")
    simulated.astype("<f8").tofile(folder / "zstats.bin")
    (folder / "meta.csv").write_text(
        f"{n_obs},{n_strategies},{n_boot},{BLOCK},{seed}\n", encoding="utf-8"
    )


def run_octave(folder: Path, mfe_bootstrap: Path) -> None:
    subprocess.run(
        command(
            [
                "octave-cli",
                "--no-gui",
                "--quiet",
                to_wsl(SCRIPTS / "octave" / "run_bsds.m"),
                to_wsl(folder),
                to_wsl(mfe_bootstrap),
                to_wsl(SCRIPTS / "octave" / "shadow"),
            ]
        ),
        check=True,
    )


def run_r(folder: Path) -> None:
    subprocess.run(
        command(["Rscript", to_wsl(SCRIPTS / "r" / "run_stepwise.R"), to_wsl(folder)]), check=True
    )


def read_bsds(folder: Path) -> dict[str, float]:
    with (folder / "bsds.csv").open(encoding="utf-8", newline="") as fh:
        return {key: float(value) for key, value in next(csv.DictReader(fh)).items()}


def read_stepwise(folder: Path) -> list[int]:
    return [int(item) for item in (folder / "stepwise.txt").read_text("utf-8").split()]


def our_side(case: Case, n_boot: int, folder: Path) -> dict[str, Any]:
    """This library on its own draws, and on the exported resamples."""
    d = case_data(case)
    seed = 3000 + case.case_id
    indices = np.fromfile(folder / "indices.bin", dtype="<i4").reshape(n_boot, d.shape[0]) - 1
    plan: NDArray[np.int64] = indices.astype(np.int64)
    own_spa = ol.spa_test(d, n_boot=n_boot, block_length=BLOCK, rng=seed)
    same_spa = ol.spa_test(d, indices=plan, block_length=BLOCK)
    row: dict[str, Any] = {
        "rc_ours": ol.reality_check(d, n_boot=n_boot, block_length=BLOCK, rng=seed + 1).pvalue,
        "rc_same_ours": ol.reality_check(d, indices=plan, block_length=BLOCK).pvalue,
    }
    for kind in KINDS:
        row[f"spa_{kind}_ours"] = getattr(own_spa, f"pvalue_{kind}")
        row[f"spa_{kind}_same_ours"] = getattr(same_spa, f"pvalue_{kind}")
    step = ol.romano_wolf_stepdown(d, indices=plan, block_length=BLOCK)
    row["stepm_ours"] = np.flatnonzero(step.rejected).tolist()
    row["all_means_negative"] = bool(d.mean(axis=0).max() <= 0.0)
    return row


def combine(
    case: Case, n_boot: int, ours: dict[str, Any], bsds: dict[str, float], stepwise: list[int]
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case": case.case_id,
        "n_strategies": case.n_strategies,
        "n_obs": case.n_obs,
        "planted": case.planted,
        "n_boot": n_boot,
        "all_means_negative": ours["all_means_negative"],
        "rc_ours": ours["rc_ours"],
        "rc_bsds": bsds["rc_u"],
        "rc_same_ours": ours["rc_same_ours"],
        "rc_same_bsds": bsds["same_rc_u"],
    }
    for kind, letter in KINDS.items():
        row[f"spa_{kind}_ours"] = ours[f"spa_{kind}_ours"]
        row[f"spa_{kind}_bsds"] = bsds[letter]
        row[f"spa_{kind}_same_ours"] = ours[f"spa_{kind}_same_ours"]
        row[f"spa_{kind}_same_bsds"] = bsds[f"same_{letter}"]
    row["stepm_ours"] = " ".join(str(i) for i in ours["stepm_ours"])
    row["stepm_stepwise"] = " ".join(str(i) for i in stepwise)
    row["stepm_match"] = ours["stepm_ours"] == stepwise
    return row


def run_case(case: Case, n_boot: int, root: Path, mfe_bootstrap: Path) -> dict[str, Any]:
    folder = root / f"case_{case.case_id:02d}"
    export_case(case, n_boot, folder)
    run_octave(folder, mfe_bootstrap)
    run_r(folder)
    ours = our_side(case, n_boot, folder)
    return combine(case, n_boot, ours, read_bsds(folder), read_stepwise(folder))


def run_all(n_boot: int, root: Path, toolbox: Path) -> list[dict[str, Any]]:
    mfe_bootstrap = ensure_toolbox(toolbox)
    workers = min(MAX_WORKERS, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(lambda case: run_case(case, n_boot, root, mfe_bootstrap), fixed_cases())
        )


INDEPENDENT = ("rc_ours", "rc_bsds"), *((f"spa_{k}_ours", f"spa_{k}_bsds") for k in KINDS)
IDENTICAL = (
    ("rc_same_ours", "rc_same_bsds"),
    *((f"spa_{k}_same_ours", f"spa_{k}_same_bsds") for k in KINDS),
)


def _largest_gap(rows: Sequence[dict[str, Any]], pairs: Sequence[tuple[str, str]]) -> float:
    return float(max((abs(r[a] - r[b]) for r in rows for a, b in pairs), default=0.0))


def check_criteria(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """AC-W-07's limits on every case, and the same gaps without the floor cases.

    ``bsds.m`` has no floor at zero, so it returns p = 1 when every mean is negative. The figures
    without those cases show what the floor explains. They do not change the verdict.
    """
    rest = [row for row in rows if not row["all_means_negative"]]
    worst = _largest_gap(rows, INDEPENDENT)
    same = _largest_gap(rows, IDENTICAL)
    matches = sum(row["stepm_match"] for row in rows)
    return {
        "pvalues_within_tolerance": bool(worst <= PVALUE_TOLERANCE),
        "max_pvalue_diff": worst,
        "identical_indices_agree": bool(same <= IDENTICAL_TOLERANCE),
        "max_identical_diff": same,
        "stepm_sets_match": bool(matches >= 19),
        "stepm_matches": int(matches),
        "floor_cases": [row["case"] for row in rows if row["all_means_negative"]],
        "max_pvalue_diff_without_floor_cases": _largest_gap(rest, INDEPENDENT),
        "max_identical_diff_without_floor_cases": _largest_gap(rest, IDENTICAL),
    }


def write_report(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in row.items()})
