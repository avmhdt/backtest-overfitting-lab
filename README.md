# Backtest overfitting study

> Implement the statistical tools that tell a real signal from a lucky one, then turn them on your own work.

**Status:** planning

| | |
|---|---|
| Track | Quant Researcher |
| Estimated effort | 3–5 weeks |
| Stack | Python (NumPy, Numba, SciPy) |
| Data cost | Free |
| Strongest for | Any QR seat; pairs with every other project |

## Requirements

- [x] Purged k-fold with embargo and combinatorial purged CV (CPCV), handling overlapping multi-day labels. (FR-08, FR-09)
- [x] Probabilistic and Deflated Sharpe Ratio (Bailey and López de Prado), adjusting for skew, kurtosis and the number and variance of trials. (FR-03, FR-05)
- [x] Probability of Backtest Overfitting via CSCV (Bailey, Borwein, López de Prado and Zhu). (FR-10)
- [x] Data-snooping tests: White’s Reality Check, Hansen’s SPA, and the Romano–Wolf stepdown. (FR-12, FR-13, FR-14)
- [x] Minimum backtest length and a Sharpe ratio haircut calculator (Harvey and Liu). (FR-06, FR-07)
- [x] A strategy factory that generates thousands of parameter variants (crossovers, breakouts, mean reversion) to test against. (FR-15)
- [x] A synthetic data generator with known ground truth: pure noise, and noise with a planted signal of chosen strength. (FR-16)

## Data

Raw and licensed data live in `data/`, which is git-ignored. Only download scripts and small synthetic fixtures are committed.

| Dataset | Cost | What it gives you |
|---|---|---|
| Synthetic returns (GBM, block bootstrap) | Free | Ground truth for validating every test. Generated in your own code. |
| [Stooq bulk daily data](https://stooq.com/db/h/) | Free | Decades of daily prices for indices, ETFs, futures and FX. |
| [AQR datasets](https://www.aqr.com/Insights/Datasets) | Free | Long factor histories for multiple-testing case studies. |
| [Open Source Asset Pricing](https://www.openassetpricing.com) | Free | A real ‘many signals’ universe to apply corrections to. |

## Success criteria

### Works (correct and reproducible)

- [x] On pure noise, the best of 1,000 strategies shows a high naive Sharpe while the DSR is not significant and PBO is near or above 0.5. (AC-W-01)
- [ ] On planted-signal data, each test detects the signal at roughly its theoretical power.
- [x] Implementations reproduce the numerical examples in the original papers to three decimals. (AC-W-03, with G-03c as a documented exception)
- [x] A simulation shows the accuracy gap between standard k-fold and purged k-fold on overlapping labels. (AC-W-04)

### Impresses hirers

- [ ] Power curves: how large a true Sharpe, and how long a backtest, each test needs before it detects a signal.
- [ ] A table of naive vs deflated Sharpe for your own projects ([equity-factor-pipeline](https://github.com/avmhdt/equity-factor-pipeline), [llm-text-alpha](https://github.com/avmhdt/llm-text-alpha), [stat-arb-pairs](https://github.com/avmhdt/stat-arb-pairs)).
- [ ] Packaged as a pip-installable library with docs and an interactive Sharpe-haircut page.
- [ ] A short paper-style writeup a PM could read in ten minutes.

## Questions this project must answer

- Why does the DSR use the variance of trial Sharpes, not just the count?
- When is PBO misleading?
- How do you count trials honestly when you iterated by hand?
