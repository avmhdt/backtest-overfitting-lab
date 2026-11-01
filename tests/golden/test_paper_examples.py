from pathlib import Path

import pytest

from overfitlab_study.validation.papers import evaluate_case, load_cases

CASES = load_cases(Path(__file__).parent / "paper_examples.yaml")
SOURCED = [
    pytest.param(
        case,
        id=case["id"],
        marks=[pytest.mark.xfail(strict=True)] if case.get("known_discrepancy") else [],
    )
    for case in CASES
    if case["status"] == "sourced"
]


@pytest.mark.golden
@pytest.mark.parametrize("case", SOURCED)
def test_printed_value(case):
    result = evaluate_case(case)
    assert result.passed, f"{case['id']}: printed {result.printed}, computed {result.computed}"


@pytest.mark.golden
def test_every_group_is_present():
    assert {case["group"] for case in CASES} == {f"G-0{i}" for i in range(1, 9)}
