"""The docs site against the code: public names, docstrings, theory pages, examples."""

from __future__ import annotations

import csv
import inspect
import re
import runpy
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import overfitlab

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs"


def _public_functions() -> list[str]:
    return [name for name in overfitlab.__all__ if inspect.isfunction(getattr(overfitlab, name))]


def test_api_page_lists_every_public_name() -> None:
    text = (DOCS / "api.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"^\s+- (\w+)$", text, flags=re.MULTILINE))
    listed |= set(re.findall(r"`overfitlab\.(\w+)`", text))
    assert listed == set(overfitlab.__all__)


@pytest.mark.parametrize("name", _public_functions())
def test_public_function_docstring_has_the_numpy_sections(name: str) -> None:
    doc = inspect.getdoc(getattr(overfitlab, name)) or ""
    for section in ("Parameters", "Returns", "Examples"):
        assert re.search(rf"^{section}\n-+$", doc, flags=re.MULTILINE), section


@pytest.mark.parametrize(
    "name",
    [
        "probabilistic_sharpe_ratio",
        "psr_from_returns",
        "min_track_record_length",
        "expected_max_sharpe",
        "deflated_sharpe_ratio",
        "dsr_from_returns",
        "min_backtest_length",
        "haircut_sharpe",
        "assemble_paths",
        "pbo_cscv",
        "optimal_block_length",
        "stationary_bootstrap_indices",
        "reality_check",
        "spa_test",
        "romano_wolf_stepdown",
    ],
)
def test_method_docstring_cites_its_paper(name: str) -> None:
    assert re.search(r"\((19|20)\d\d", inspect.getdoc(getattr(overfitlab, name)) or "")


@pytest.mark.parametrize("page", ["psr-dsr", "pbo", "trial-counting"])
def test_theory_page_is_short_with_a_figure_on_disk(page: str) -> None:
    path = DOCS / "theory" / f"{page}.md"
    text = path.read_text(encoding="utf-8")
    assert len(text.split()) <= 600
    assert text.count("**The question:") == 1
    figures = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert figures
    assert all((path.parent / figure).is_file() for figure in figures)


def test_getting_started_examples_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    text = (DOCS / "getting-started.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    assert len(blocks) == 4
    for block in blocks:
        exec(block, {})  # noqa: S102
    out = capsys.readouterr().out
    assert "0.9004" in out
    assert "FAIL" in out


def test_import_stays_light() -> None:
    probe = (
        "import sys, overfitlab\n"
        "heavy = {'pyarrow', 'typer', 'polars', 'arch', 'sklearn', 'matplotlib'}\n"
        "print(sorted(heavy & {name.split('.')[0] for name in sys.modules}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_readme_quickstart_runs() -> None:
    smoke = runpy.run_path(str(ROOT / "scripts" / "smoke_install.py"))
    assert smoke["run_quickstart"](ROOT / "README.md") == "0.9004"


def test_version_is_the_same_in_both_places() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == overfitlab.__version__


def test_memo_keeps_to_its_limits() -> None:
    check = runpy.run_path(str(ROOT / "scripts" / "check_memo.py"))
    assert check["problems"]((DOCS / "memo.md").read_text(encoding="utf-8")) == []


def _table(name: str) -> list[dict[str, str]]:
    path = ROOT / "reports" / "tables" / f"{name}.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize("path", [ROOT / "README.md", DOCS / "memo.md"])
def test_prose_quotes_the_current_tables(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    noise = {row["setting"]: row for row in _table("noise_experiment")}
    stooq = _table("case_stooq")[0]
    osap = {row["window"]: row for row in _table("case_osap")}
    projects = [row for row in _table("own_projects") if row["verdict"]]
    expected = [
        f"{float(noise['independent']['median_best_sr_annual']):.2f}",
        f"{float(noise['factory_gbm']['median_best_sr_annual']):.2f}",
        f"Sharpe {float(stooq['naive_sr_annual']):.2f}",
        f"DSR {float(stooq['dsr']):.2f}",
        f"PBO {float(stooq['pbo']):.2f}",
        f"N = {float(projects[0]['n_raw_project']):.0f}"
        if path.name == "README.md"
        else f"{float(projects[0]['n_raw_project']):.0f} trials",
        f"{osap['post_publication']['t_above_1_96']} ",
        f"{float(next(row for row in projects if row['family'] == 'investment')['dsr_project_n']):.3f}",
    ]
    for value in expected:
        assert value in text, value
    if path.name == "README.md":
        for row in projects:
            cells = (
                f"| {row['family']} | {float(row['n_raw']):.0f} | "
                f"{float(row['naive_sr_annual']):.2f} | {float(row['sr0_annual']):.2f} | "
                f"{float(row['dsr']):.3f} | {row['verdict']} | "
                f"{float(row['dsr_project_n']):.3f} |"
            )
            assert cells in text, cells
