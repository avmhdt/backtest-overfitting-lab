# Validation

Every value below is transcribed from the paper itself, with its page, table or exhibit in `tests/golden/paper_examples.yaml`. `olab-study validate papers` writes the comparison to `reports/validation/paper_examples.csv`.

## Paper examples

| Group | Source | Status |
|---|---|---|
| G-01 | Bailey and López de Prado (2014), DSR example | Passes, plus the paper's N = 46 and Normal-returns cases |
| G-02 | Bailey et al. (2014), MinBTL | Passes at printed precision. The paper prints "5 years"; the computed value is in the report |
| G-03 | Bailey and López de Prado (2012), PSR example | Two of three pass; see below |
| G-04 | Bailey and López de Prado (2012), MinTRL | Passes |
| G-05 | Harvey and Liu (2015), Exhibit 5 | Passes |
| G-06 | Lo (2002), Table 2 | Passes on twelve selected entries |
| G-07 | López de Prado (2018), CPCV counts | Pending source: the book is not available to the build agent |
| G-08 | Harvey, Liu and Zhu (2016) | Passes on the Bonferroni benchmark and on the p-value of the 3.0 hurdle |

## Known discrepancies

**G-03c, PSR with three years of monthly data.** The paper prints 0.953. The computed value from the paper's printed input SR = 0.458 is 0.95351, which misses the 5e-4 tolerance by 5e-6. The tolerance was not changed. The most likely cause is rounding of the input: the paper feeds a value printed to three decimals. Figure 6 also prints an annual Sharpe ratio of 1.585, which gives a different rounded input, and `G-03c-alt` shows that one passes. That row is reported alongside G-03c and does not replace it.

**Exhibit 7 against the reference code.** Harvey and Liu's Exhibit 7 prints p₀ = 0.396, 0.444, 0.485, 0.601 and 0.840, while their reference `Haircut_SR.m` uses 0.3966, 0.4459, 0.4860, 0.5990 and 0.8390. This library uses the reference code's values, because Exhibit 5 was produced by that code.

**Simulated Holm and BHY values.** The authors ran 2,000 unseeded simulations, so a printed value is one random draw. Tolerances for G-05e to G-05j are 3 standard deviations of a 2,000-run median. They were measured from this implementation's spread and fixed in the YAML before any comparison with the printed values. The default here is 5,000 runs, as in the paper's appendix.

**Reference code details kept deliberately.** Bonferroni uses `n_tests` while Holm and BHY use `n_tests + 1`. BHY leaves the largest p-value unscaled. The reference code annualises daily data with 360 days; here `periods_per_year` is always explicit.

## Methods with no printed example

White's Reality Check, Hansen's SPA and the Romano–Wolf stepdown are validated against the independent `arch` package. PBO is validated against a hand-computed case and the qualitative Monte Carlo results of Bailey et al. (2017). Both arrive in M4.
