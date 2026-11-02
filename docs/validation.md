# Validation

Every value below is transcribed from the paper itself, with its page, table or exhibit in `tests/golden/paper_examples.yaml`. `olab-study validate papers` writes the comparison to `reports/validation/paper_examples.csv`.

## Paper examples

| Group | Source | Status |
|---|---|---|
| G-01 | Bailey and López de Prado (2014), DSR example | Passes, plus the paper's N = 46 and Normal-returns cases |
| G-02 | Bailey et al. (2014), MinBTL | Passes at printed precision. The paper prints "5 years"; the computed value is in the report |
| G-03 | Bailey and López de Prado (2012), PSR example | Two of three pass. G-03c is a documented exception; see below |
| G-04 | Bailey and López de Prado (2012), MinTRL | Passes |
| G-05 | Harvey and Liu (2015), Exhibit 5 | Passes |
| G-06 | Lo (2002), Table 2 | Passes on twelve selected entries |
| G-07 | López de Prado (2018), section 12.4.1, p. 164 | Passes: 15 splits and 5 paths for N = 6 and k = 2. Checked in the owner's copy of the book |
| G-08 | Harvey, Liu and Zhu (2016) | Passes on the Bonferroni benchmark and on the p-value of the 3.0 hurdle |

## Known discrepancies

**G-03c, PSR with three years of monthly data.** The paper prints 0.953. The computed value from the paper's printed input SR = 0.458 is 0.95351, which misses the 5e-4 tolerance by 5e-6. The tolerance was not changed. The most likely cause is rounding of the input: the paper feeds a value printed to three decimals. Figure 6 also prints an annual Sharpe ratio of 1.585, which gives a different rounded input, and `G-03c-alt` shows that one passes. That row is reported alongside G-03c and does not replace it. The owner ruled G-03c a documented exception to AC-W-03 on 2026-11-01: it still reports `pass=false`, and the tolerance stands.

**Exhibit 7 against the reference code.** Harvey and Liu's Exhibit 7 prints p₀ = 0.396, 0.444, 0.485, 0.601 and 0.840, while their reference `Haircut_SR.m` uses 0.3966, 0.4459, 0.4860, 0.5990 and 0.8390. This library uses the reference code's values, because Exhibit 5 was produced by that code.

**Simulated Holm and BHY values.** The authors ran 2,000 unseeded simulations, so a printed value is one random draw. Tolerances for G-05e to G-05j are 3 standard deviations of a 2,000-run median. They were measured from this implementation's spread and fixed in the YAML before any comparison with the printed values. The default here is 5,000 runs, as in the paper's appendix.

**Reference code details kept deliberately.** Bonferroni uses `n_tests` while Holm and BHY use `n_tests + 1`. BHY leaves the largest p-value unscaled. The reference code annualises daily data with 360 days; here `periods_per_year` is always explicit.

## Planted-signal power

`olab-study sim power --check` compares each test with the closed forms in [the power page](theory/power.md). `reports/tables/power_check.csv` has every cell.

- PSR, the Reality Check, SPA and StepM are within tolerance in all four cells, and the median PBO falls as the true Sharpe ratio rises.
- **AC-W-02 is not met.** The DSR falls short of its reference in the three cells with N > 1. The reference treats V as known, while the DSR estimates it from the same trials, and a high maximum inflates that estimate.
- The `dsr_known_v` rows fix V at 1/(T − 1), and they match the reference in all four cells. That supports the explanation. It does not close the criterion: they were added after the failure, and they are marked `diagnostic`. The owner decides what to do.

## Methods with no printed example

White's Reality Check, Hansen's SPA and the Romano–Wolf stepdown have no printed numbers, so they are checked against the independent `arch` package. PBO is checked against a hand-computed case and against the qualitative Monte Carlo results of Bailey et al. (2017).

### Cross-check with `arch`

`olab-study validate arch` runs 20 fixed cases at B = 10,000 and writes `reports/validation/arch_xcheck.csv`.

**`arch` 8 does not studentise.** Its `SPA` compares raw means, with no floor at zero, whether or not `studentize=True` is set. The flag only feeds the screening threshold of the consistent p-value. `tests/integration/test_arch_xcheck.py` proves this on data with very unequal volatility: the p-values are identical either way. Hansen's statistic is studentised and floored at zero, and that is the default here, as FR-13 asks.

So the two are not the same statistic, and the check compares like with like through `spa_test(studentize=False)`:

| Check | Requirement | Result |
|---|---|---|
| Reality Check and raw SPA p-values, independent draws | at most 0.02 | see `arch_xcheck.csv` and the experiment log |
| Same resamples as `arch` (read from a twin sampler) | at most 1e-8 | see `arch_xcheck.csv` and the experiment log |
| StepM rejected sets, studentised here and raw in `arch` | at least 19 of 20 | see `arch_xcheck.csv` and the experiment log |

The gap between Hansen's studentised SPA and `arch` is reported separately as `max_studentised_gap`. It is large on some null cases, and it should be: it measures a real difference between the two statistics.

**AC-W-07 stays open.** These checks are evidence that the bootstrap and the recentring agree with `arch`. They say nothing about the studentised default, and the owner ruled on 2026-11-01 that they do not close the criterion.

### Candidates for a studentised reference

Each was read at source on 2026-11-01. None has been run, because none of the runtimes is installed here.

| Candidate | Studentises? | Matches Hansen? | Access and licence | Assessment |
|---|---|---|---|---|
| `bsds.m`, MFE Toolbox (Sheppard) | Yes, it divides by the long-run standard deviation | Same kernel, the three recentrings and the log-log rule. It has no floor at zero on the statistic | Free. GNU Octave plus a clone of `bashtage/mfe-toolbox`. No licence file, only a copyright notice, so use it locally and never vendor it | **Recommended for SPA and the Reality Check** |
| `StepwiseTest`, R (Hsu, Kuan and Yen) | Takes whatever statistics it is given | `FWERkControl` with k = 1 is the stepdown, so it checks the step logic only | Free. R, CRAN, GPL ≥ 2 | **Recommended for StepM**, fed the statistics |
| `ForecastEval.jl` (Bowers) | Yes | Floor and recentrings match, but the variance is an Epanechnikov HAC estimator, as its own docs state | Free. Julia, MIT. Last commit 2021 | Second opinion only, since the variance differs by design |
| MulCom 3.0, Ox (Hansen, Lunde and Laurent) | Not verified: the page does not say and the manual was not read | Hansen's own code, the strongest authority | The free Ox Console covers academic research and teaching only, and this project is neither, so it probably needs a paid licence, which is not approved. MulCom's own page states no terms | Needs the owner to check eligibility |
| `RCtest`, R | Yes | No: moving-block bootstrap, and no lower, consistent and upper p-values in Hansen's sense | Free. R, GPL-3 | Rejected |

Proposed check, with the limits unchanged: the same 20 cases at B = 10,000 and block length 5, comparing `spa_test` with `bsds(..., 'STUDENTIZED')` on all three p-values at 0.02. For identical indices at 1e-8, a `stationary_bootstrap.m` placed ahead on the Octave path would return the indices of this library, with `bsds.m` itself unedited. StepM rejected sets would go through `FWERkControl`, at 19 of 20.

**One known difference.** `bsds.m` has no floor at zero, so when every mean is negative it returns p = 1, where Hansen's floor gives a smaller value. One of the 20 cases is like this: case 2, where this library gives 0.716, 0.969 and 0.969. That case would exceed 0.02 for that reason alone, and how to treat it is the owner's call. `bsds.m` also shares an author with `arch`, though it is separate code that does studentise.

### PBO by hand

`tests/fixtures/pbo_s4.csv` has three trials over four blocks, with the mean as the metric. There are six ways to choose two in-sample blocks. Ranks run from 1, the worst, to 3, the best.

| In-sample blocks | In-sample means (a, b, c) | Winner | Out-of-sample means | Rank | ω = rank/4 | λ |
|---|---|---|---|---|---|---|
| 0, 1 | 4.5, 0.5, 1.5 | a | 2.5, 1.5, 3.5 | 2 | 0.50 | 0 |
| 0, 2 | 4.0, 1.5, 0.5 | a | 3.0, 0.5, 4.5 | 2 | 0.50 | 0 |
| 0, 3 | 3.5, 1.0, 3.0 | a | 3.5, 1.0, 2.0 | 3 | 0.75 | ln 3 |
| 1, 2 | 3.5, 1.0, 2.0 | a | 3.5, 1.0, 3.0 | 3 | 0.75 | ln 3 |
| 1, 3 | 3.0, 0.5, 4.5 | c | 4.0, 1.5, 0.5 | 1 | 0.25 | −ln 3 |
| 2, 3 | 2.5, 1.5, 3.5 | c | 4.5, 0.5, 1.5 | 2 | 0.50 | 0 |

Four of the six logits are at or below zero, so PBO = 4/6. `tests/unit/test_pbo.py::test_hand_case_with_four_blocks` asserts this exactly.
