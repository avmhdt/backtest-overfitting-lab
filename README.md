# Backtest overfitting study

> The statistics that tell a real trading signal from a lucky one, packaged as a library and turned on real research.

`overfitlab` deflates a backtest for the search behind it. Run 1,000 strategies on pure noise and the best shows a median annual Sharpe ratio of 1.43. It looks publishable, and it's nothing. This project implements the tests that catch that, checks them against the papers and simulations, and applies them to public data and to the owner's own trial logs.

The [ten-minute memo](https://avmhdt.github.io/backtest-overfitting-lab/memo/) gives the argument. The [docs site](https://avmhdt.github.io/backtest-overfitting-lab/) has the API, the theory and a [calculator](https://avmhdt.github.io/backtest-overfitting-lab/calculator/).

## Quickstart

```bash
pip install overfitlab
```

```python
import overfitlab as ol

# A 2.5 annual Sharpe ratio over five years, the best of 100 trials.
sr = 2.5 / 250**0.5
result = ol.deflated_sharpe_ratio(
    sr, 1250, n_trials=100, var_trials=0.5 / 250, skew=-3.0, kurtosis=10.0
)
print(round(result.dsr, 4))  # 0.9004
```

That Sharpe ratio misses the 0.95 bar once its 100 trials and fat left tail are counted. [Getting started](https://avmhdt.github.io/backtest-overfitting-lab/getting-started/) covers trial logs, cross-validation and the verdict gate.

## The problem

Researchers run many backtests and report the best. The best of many noisy trials has a high Sharpe ratio even when no trial has an edge, and a backtest report doesn't show how many trials stood behind it. The hypothesis here is that a small set of published corrections, applied with an honest trial count, separates luck from signal, and that each has failure modes worth measuring.

## Data

Raw and licensed data live in `data/`, which is git-ignored. Only download scripts, aggregates and small synthetic fixtures are committed.

| Dataset | Cost | What it gives |
|---|---|---|
| Synthetic returns (GBM, block bootstrap) | Free | Ground truth for validating every test. |
| [Stooq bulk daily data](https://stooq.com/db/h/) and Sharadar | Free, and the owner's bundle | Daily prices for 12 ETFs, 2006 to 2026. |
| [Ken French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and [AQR datasets](https://www.aqr.com/Insights/Datasets) | Free | 56 long factor histories for multiple-testing case studies. |

AQR data: Century of Factor Premia, Time Series Momentum, Betting Against Beta and Quality Minus Junk, from [AQR Capital Management](https://www.aqr.com/Insights/Datasets), used with AQR's written permission. This repository publishes aggregate statistics and charts only, not AQR's files or series.
| [Open Source Asset Pricing](https://www.openassetpricing.com) | Free | A real "many signals" universe of 212 predictors, release 2026.10. |

## Method

| Method | Question it answers |
|---|---|
| PSR and MinTRL | Is this Sharpe ratio real, given its length and return shape? |
| DSR and MinBTL | Is it real, given how many trials produced it? |
| PBO via CSCV | Does in-sample selection carry over out of sample? |
| Reality Check, SPA, StepM | Does anything in this set beat the benchmark, and which? |
| Harvey–Liu haircut | What is the Sharpe ratio worth after multiple testing? |
| Purged k-fold and CPCV | How do I cross-validate overlapping labels without leaking? |

A verdict gate requires the DSR, PBO, SPA, StepM and the haircut to pass together. Trial logs follow a frozen [contract](https://avmhdt.github.io/backtest-overfitting-lab/trial-log/), so N and V come from a record and not from memory.

Each method is checked three ways: against the papers' printed examples, against closed-form power, and against independent implementations. [Validation](https://avmhdt.github.io/backtest-overfitting-lab/validation/) has the details.

## Results

| Experiment | Result |
|---|---|
| Best of 1,000 on noise, 100 seeds | Median best Sharpe 1.43. Naive PSR significant in 100 seeds, DSR in 0. Median PBO 0.50. |
| 1,034 factory variants on noise | Median best Sharpe 1.13. DSR significant in 0 seeds. Median PBO 0.52. |
| Leakage on noise, 20-day labels | Shuffled k-fold accuracy 0.703, purged 0.495. Gap 0.208. |
| Factory on 12 ETFs, 1,034 trials | Sharpe 0.76 against a bar of 0.83. DSR 0.39, PBO 0.31, verdict FAIL. Holdout Sharpe 0.68. |
| 56 factors from Ken French and AQR | 30 survive StepM. |
| 212 OSAP predictors, full history | 173 at t > 1.96, 134 at t > 3, 149 after BHY, 96 after StepM. |
| 208 OSAP predictors, after publication | 54 at t > 1.96, 24 at t > 3, 9 after BHY, 8 after StepM. |

![The best of many noise strategies against its DSR](https://avmhdt.github.io/backtest-overfitting-lab/assets/figures/noise_best_vs_dsr.png)

Detection takes a long record. With a true Sharpe ratio of 1 among 100 trials, the Reality Check, SPA and StepM need 20 years for 80% power, and the DSR doesn't reach it within the 20-year grid. See the [power curves](https://avmhdt.github.io/backtest-overfitting-lab/theory/power/).

### The owner's own trials

Only one of the six projects, `equity-factor-pipeline`, has a trial log so far.

| Family | Own N | Sharpe | SR₀ | DSR | Verdict | DSR at N = 613 |
|---|---:|---:|---:|---:|---|---:|
| composite | 305 | 1.25 | 2.34 | 0.000 | FAIL | 0.000 |
| investment | 32 | 1.00 | 0.70 | 0.952 | PASS | 0.429 |
| low_risk | 32 | 0.80 | 0.45 | 0.957 | PASS | 0.734 |
| momentum | 66 | 0.60 | 0.73 | 0.261 | FAIL | 0.036 |
| quality | 44 | 0.98 | 0.63 | 0.951 | PASS | 0.678 |
| replication | 86 | 0.81 | 0.60 | 0.840 | FAIL | 0.595 |
| size | 8 | 0.09 | 0.05 | 0.588 | FAIL | 0.469 |
| value | 40 | 0.68 | 0.51 | 0.802 | FAIL | 0.412 |

Three families pass when each is judged alone. They were not found alone: the project ran 613 trials. At that count, no family reaches 0.95. `investment` fails outright, because the bar of 1.03 is above its Sharpe ratio of 1.00. Which count applies depends on whether the families were independent ideas or one search, so this table can't settle whether they're real. It does show that the verdict turns on that choice.

## What failed

- **DSR power (AC-W-02).** The DSR rejects less often than its reference: 0.036 against 0.168, 0.149 against 0.207, and 0.003 against 0.056 at the three cells that miss. V is estimated from the same trials as the planted winner, which raises its own bar. A version given the true V matches. This is kept as a finding, and no limit was changed.
- **G-03c.** One of the 40 paper examples misses its tolerance by 5e-6. It's a documented exception.
- **Studentised SPA check (AC-W-07).** Against `bsds.m`, 19 of 20 cases agree within 0.013, and StepM agrees in all 20. Case 2 has no positive mean, and `bsds.m` has no floor there, so it reports p = 1.
- **Own-projects coverage (AC-H-02).** One of six projects has a trial log.
- **Holm and BHY on the calculator (FR-34).** Deferred. The page gives the exact Bonferroni haircut.

## Limitations

- A trial log records what was run, not what was thought about, so N is a lower bound.
- The tests here compare with zero. A long-only ETF strategy can beat zero on the equity premium alone, which is why the factory case passes SPA and StepM and still fails the DSR.
- Fewer predictors survive after publication partly because those samples are shorter, so that drop isn't all decay.
- PBO measures whether rankings persist, not whether money is made. An edge that died halfway through scores near 0. See [when PBO misleads](https://avmhdt.github.io/backtest-overfitting-lab/theory/pbo/).
- Power curves assume independent strategies with one planted signal.

## How to reproduce

```bash
uv sync
uv run olab-study all
```

The full run takes about an hour and regenerates every table and figure from the data in `data/`. Without that data, the real-data steps are skipped with a message. `--quick` runs a reduced version in about two minutes. Each run adds rows to `results/experiment_log.csv`.

## Requirements

- [x] Purged k-fold with embargo and combinatorial purged CV (CPCV), handling overlapping multi-day labels. (FR-08, FR-09)
- [x] Probabilistic and Deflated Sharpe Ratio (Bailey and López de Prado), adjusting for skew, kurtosis and the number and variance of trials. (FR-03, FR-05)
- [x] Probability of Backtest Overfitting via CSCV (Bailey, Borwein, López de Prado and Zhu). (FR-10)
- [x] Data-snooping tests: White's Reality Check, Hansen's SPA, and the Romano–Wolf stepdown. (FR-12, FR-13, FR-14)
- [x] Minimum backtest length and a Sharpe ratio haircut calculator (Harvey and Liu). (FR-06, FR-07, FR-33)
- [x] A strategy factory that generates thousands of parameter variants (crossovers, breakouts, mean reversion) to test against. (FR-15)
- [x] A synthetic data generator with known ground truth: pure noise, and noise with a planted signal of chosen strength. (FR-16)

## Success criteria

### Correct and reproducible

- [x] On pure noise, the best of 1,000 strategies shows a high naive Sharpe while the DSR is not significant and PBO is near or above 0.5. (AC-W-01)
- [ ] On planted-signal data, each test detects the signal at roughly its theoretical power. (AC-W-02: met for PSR, RC, SPA and StepM, missed for the DSR)
- [x] Implementations reproduce the numerical examples in the original papers to three decimals. (AC-W-03, with G-03c as a documented exception)
- [x] A simulation shows the accuracy gap between standard k-fold and purged k-fold on overlapping labels. (AC-W-04)

### Depth

- [x] Power curves: how large a true Sharpe, and how long a backtest, each test needs before it detects a signal. (AC-H-01)
- [ ] A table of naive vs deflated Sharpe for your own projects ([equity-factor-pipeline](https://github.com/avmhdt/equity-factor-pipeline), [llm-text-alpha](https://github.com/avmhdt/llm-text-alpha), [stat-arb-pairs](https://github.com/avmhdt/stat-arb-pairs)). (AC-H-02: one of six projects so far)
- [x] Packaged as a pip-installable library with docs and an interactive Sharpe-haircut page. (AC-H-03)
- [x] A short paper-style writeup a PM could read in ten minutes. (AC-H-04)

## Questions this project must answer

- [Why does the DSR use the variance of trial Sharpes, not just the count?](https://avmhdt.github.io/backtest-overfitting-lab/theory/psr-dsr/)
- [When is PBO misleading?](https://avmhdt.github.io/backtest-overfitting-lab/theory/pbo/)
- [How do you count trials honestly when you iterated by hand?](https://avmhdt.github.io/backtest-overfitting-lab/theory/trial-counting/)
