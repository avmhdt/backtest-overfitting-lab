# overfitlab: backtest overfitting statistics

> The statistics that tell a real trading signal from a lucky one, packaged as a library and turned on real research.

`overfitlab` deflates a backtest for the search behind it. Run 1,000 strategies on pure noise and the best shows a median annual Sharpe ratio of 1.43. It looks publishable, and it's nothing. This project implements the tests that catch that, checks them against the papers and simulations, and applies them to public data and to a real research project's trial log.

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
| [Stooq bulk daily data](https://stooq.com/db/h/) and Sharadar | Stooq is free, Sharadar is a paid subscription | Daily prices for 12 ETFs, 2006 to 2026. |
| [Ken French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and [AQR datasets](https://www.aqr.com/Insights/Datasets) | Free | 56 long factor histories for multiple-testing case studies. |
| [Open Source Asset Pricing](https://www.openassetpricing.com) | Free | A real "many signals" universe of 212 predictors, release 2026.10. |

AQR data: Century of Factor Premia, Time Series Momentum, Betting Against Beta and Quality Minus Junk, from [AQR Capital Management](https://www.aqr.com/Insights/Datasets), used with AQR's written permission. This repository publishes aggregate statistics and charts only, not AQR's files or series.

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

### A real trial log

On a real research project's trial log, three strategy families pass when each is judged on its own trials, and none passes at the project's whole count of 613 trials. [Trial logs in practice](https://avmhdt.github.io/backtest-overfitting-lab/projects/) has the table.

## What failed

- **DSR power.** The DSR rejects less often than its reference: 0.036 against 0.168, 0.149 against 0.207, and 0.003 against 0.056 at the three cells that miss. V is estimated from the same trials as the planted winner, which raises its own bar. A version given the true V matches. This is kept as a finding, and no limit was changed.
- **One paper example.** One of the 40 paper examples misses its tolerance by 5e-6, and is kept as a documented exception.
- **Studentised SPA check.** Against `bsds.m`, 19 of 20 cases agree within 0.013, and StepM agrees in all 20. Case 2 has no positive mean, and `bsds.m` has no floor there, so it reports p = 1.
- **Holm and BHY on the calculator.** Deferred. The page gives the exact Bonferroni haircut.

## Limitations

- A trial log records what was run, not what was thought about, so N is a lower bound.
- The tests here compare with zero. A long-only ETF strategy can beat zero on the equity premium alone, which is why the factory case passes SPA and StepM and still fails the DSR.
- Fewer predictors survive after publication partly because those samples are shorter, so that drop isn't all decay.
- PBO measures whether rankings persist, not whether money is made. An edge that died halfway through scores near 0. See [when PBO misleads](https://avmhdt.github.io/backtest-overfitting-lab/theory/pbo/).
- Power curves assume independent strategies with one planted signal.

## How to reproduce

The study runs from a clone of the [GitHub repository](https://github.com/avmhdt/backtest-overfitting-lab), because the `olab-study` commands aren't part of the PyPI package.

```bash
uv sync
uv run olab-study all
```

The full run takes about an hour and regenerates every table and figure from the data in `data/`. Without that data, the real-data steps are skipped with a message. `--quick` runs a reduced version in about two minutes. Each run adds rows to `results/experiment_log.csv`.

## Further reading

- [Why does the DSR use the variance of trial Sharpes, not just the count?](https://avmhdt.github.io/backtest-overfitting-lab/theory/psr-dsr/)
- [When is PBO misleading?](https://avmhdt.github.io/backtest-overfitting-lab/theory/pbo/)
- [How do you count trials honestly when you iterated by hand?](https://avmhdt.github.io/backtest-overfitting-lab/theory/trial-counting/)
