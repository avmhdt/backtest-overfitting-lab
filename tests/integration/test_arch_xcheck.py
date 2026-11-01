import numpy as np
import pytest
from arch.bootstrap import SPA

from overfitlab_study.validation import arch_xcheck

N_BOOT = 10_000


def test_twenty_fixed_cases():
    cases = arch_xcheck.fixed_cases()
    assert len(cases) == 20
    assert {(c.n_strategies, c.n_obs, c.planted) for c in cases} == {
        (n, t, p) for n in (5, 50) for t in (250, 1000) for p in (False, True)
    }


def test_case_data_is_fixed():
    case = arch_xcheck.fixed_cases()[3]
    assert np.array_equal(arch_xcheck.case_data(case), arch_xcheck.case_data(case))


def test_passed_needs_every_criterion():
    good = {
        "pvalues_within_tolerance": True,
        "stepm_sets_match": True,
        "identical_indices_agree": True,
    }
    assert arch_xcheck.passed(good)
    assert not arch_xcheck.passed({**good, "stepm_sets_match": False})


def test_report_is_written(tmp_path):
    rows = [{"case": 0, "gap": 0.123456789, "stepm_match": True}]
    target = tmp_path / "x" / "arch_xcheck.csv"
    arch_xcheck.write_report(rows, target)
    assert target.read_text("utf-8").splitlines() == ["case,gap,stepm_match", "0,0.123457,True"]


@pytest.mark.xcheck
def test_arch_studentize_flag_changes_nothing():
    # Why the p-values are compared through studentize=False: see docs/validation.md.
    d = np.random.default_rng(0).normal(0.0, 1.0, (500, 20)) * np.linspace(0.2, 5.0, 20)
    pvalues = []
    for flag in (True, False):
        spa = SPA(np.zeros(500), -d, block_size=5, reps=500, studentize=flag, seed=7)
        spa.compute()
        pvalues.append(spa.pvalues.to_dict())
    assert pvalues[0] == pvalues[1]


@pytest.mark.xcheck
def test_agreement_with_arch():
    rows = arch_xcheck.run_all(N_BOOT)
    criteria = arch_xcheck.check_criteria(rows)
    assert arch_xcheck.passed(criteria), criteria
