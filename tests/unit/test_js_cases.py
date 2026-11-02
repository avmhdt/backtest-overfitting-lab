"""The calculator's fixture stays in step with the Python implementation."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def test_fixture_matches_the_generator() -> None:
    build_cases = runpy.run_path(str(ROOT / "scripts" / "gen_js_cases.py"))["build_cases"]
    expected = build_cases()
    actual = json.loads((ROOT / "tests" / "fixtures" / "js_cases.json").read_text(encoding="utf-8"))
    assert [case["id"] for case in actual] == [case["id"] for case in expected]
    for found, wanted in zip(actual, expected, strict=True):
        assert found["inputs"] == pytest.approx(wanted["inputs"], rel=1e-12, abs=1e-12)
    assert [case["expected"] for case in actual] == pytest.approx(
        [case["expected"] for case in expected], rel=1e-12, abs=1e-12
    )
