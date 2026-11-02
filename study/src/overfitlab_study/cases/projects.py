"""Naive against deflated Sharpe ratios for the owner's own projects (FR-29)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import yaml

import overfitlab as ol
from overfitlab_study.reports import GRID, MUTED, SERIES_1, SERIES_2, SURFACE, write_table
from overfitlab_study.settings import Settings

PROJECTS = Path("experiments/projects.yaml")
BLANK = {
    "family": None,
    "naive_sr_annual": None,
    "n_raw": None,
    "n_clustered": None,
    "var_trials_annual": None,
    "sr0_annual": None,
    "dsr": None,
    "pbo": None,
    "bhy_haircut_sr_annual": None,
    "verdict": None,
}


def project_dir(raw: str, settings: Settings) -> Path:
    """Replace ``${OLAB_..._DIR}`` with the setting of that name."""
    name = raw.strip("${}").lower()
    return Path(getattr(settings, name))


def load_projects(path: Path = PROJECTS) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cast(list[dict[str, Any]], data["projects"])


def family_row(
    name: str, log_path: Path, family: str, selected: str | None, status: str
) -> dict[str, Any]:
    report = ol.evaluate_trial_log(log_path, families=[family], selected=selected)
    clustered: float | None = None
    try:
        log = ol.read_trial_log(log_path).filter(families=[family])
        clustered = log.n_trials(mode="clustered")
    except ValueError:
        clustered = None
    return {
        "project": name,
        "status": status,
        "family": family,
        "naive_sr_annual": report.sr_annual,
        "n_raw": report.n_trials,
        "n_clustered": clustered,
        "var_trials_annual": report.variance_annual,
        "sr0_annual": report.sr0_annual,
        "dsr": report.dsr,
        "pbo": report.pbo,
        "bhy_haircut_sr_annual": report.haircut_sr_annual,
        "verdict": report.verdict,
    }


def project_rows(project: Mapping[str, Any], settings: Settings) -> list[dict[str, Any]]:
    name = project["name"]
    log_path = project_dir(project["dir"], settings) / project["trial_log"]
    if not log_path.exists():
        return [{"project": name, "status": "no trial log", **BLANK}]
    issues = ol.validate_trial_log(log_path)
    errors = sorted({issue.code for issue in issues if issue.severity == "error"})
    if errors:
        return [{"project": name, "status": ", ".join(errors), **BLANK}]
    status = "valid"
    log = ol.read_trial_log(log_path)
    families: Sequence[str] = project["families"] or sorted(
        {str(row["strategy_family"]) for row in log.rows}
    )
    rows: list[dict[str, Any]] = []
    for family in families:
        try:
            rows.append(family_row(name, log_path, family, project["selected"], status))
        except ValueError as error:
            rows.append(
                {"project": name, "status": f"not evaluated: {error}", **BLANK, "family": family}
            )
    return rows


def plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Naive against deflated bars, one pair per evaluated family."""
    done = [row for row in rows if row["naive_sr_annual"] is not None]
    if not done:
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(done) + 1.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    positions = np.arange(len(done))[::-1].astype(float)
    labels = [f"{row['project']}: {row['family']}" for row in done]
    ax.barh(
        positions + 0.18,
        [row["naive_sr_annual"] for row in done],
        height=0.34,
        color=SERIES_1,
        label="Naive annual Sharpe ratio",
    )
    ax.barh(
        positions - 0.18,
        [row["sr0_annual"] for row in done],
        height=0.34,
        color=SERIES_2,
        label="Deflation bar SR0",
    )
    ax.set_yticks(positions, labels, fontsize=8)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=MUTED, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def run(settings: Settings, config: Path = PROJECTS) -> list[dict[str, Any]]:
    rows = [row for project in load_projects(config) for row in project_rows(project, settings)]
    reports = settings.olab_reports_dir
    write_table(rows, reports / "tables" / "own_projects.csv")
    plot(rows, reports / "figures" / "own_projects_naive_vs_deflated.png")
    return rows
