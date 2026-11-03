# Project criteria

The project's original checklists, ticked with the ids of the requirements (FR) and acceptance criteria (AC) in the spec. [Decisions](decisions.md) records how each was settled.

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
