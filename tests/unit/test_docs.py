"""The docs site against the code: public names, docstrings, theory pages, examples."""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

import overfitlab

DOCS = Path(__file__).parents[2] / "docs"


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
