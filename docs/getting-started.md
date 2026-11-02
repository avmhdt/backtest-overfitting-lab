# Getting started

## Install

```bash
pip install overfitlab
```

Python 3.12 or later. `pip install overfitlab[polars]` lets array arguments take Polars series and frames.

## Deflate a Sharpe ratio

A backtest shows an annual Sharpe ratio of 2.5 over five years, the best of 100 trials. Its returns are skewed and fat-tailed.

```python
import overfitlab as ol

sr = 2.5 / 250**0.5  # per period
result = ol.deflated_sharpe_ratio(
    sr, 1250, n_trials=100, var_trials=0.5 / 250, skew=-3.0, kurtosis=10.0
)
print(round(result.dsr, 4))  # 0.9004
```

Statistics take per-period Sharpe ratios unless a name ends in `_annual`, and kurtosis is non-excess, so a normal distribution has 3.

The same case from the command line:

```bash
overfitlab dsr --sr-annual 2.5 --periods-per-year 250 --n-obs 1250 --skew -3 --kurtosis 10 --n-trials 100 --var-trials-annual 0.5
```

## Test a whole set of trials

With a `T x N` matrix of returns, one column per trial:

```python
import numpy as np
import overfitlab as ol

returns = np.random.default_rng(0).normal(0.0, 0.01, (1260, 1000))
report = ol.evaluate(returns)
print(report.verdict)  # FAIL
print(report.reasons)
```

`evaluate` applies the gate: DSR, PBO, SPA, the stepdown and the haircut must all pass.

## Cross-validate with overlapping labels

`PurgedKFold` and `CombinatorialPurgedCV` work as a scikit-learn `cv=` argument:

```python
import numpy as np
import overfitlab as ol

t1 = np.arange(1000) + 20  # each label ends 20 bars after it starts
cv = ol.PurgedKFold(5, t1=t1, embargo_pct=0.01)
for train, test in cv.split(np.zeros(1000)):
    pass
```

## Log your trials

```python
import overfitlab as ol

trials = ol.TrialLogger("results/trial_log.csv", "results/runs")
run_id = trials.log(
    strategy_family="ma_cross",
    params={"fast": 5, "slow": 60},
    dates=["2024-01-02", "2024-01-03", "2024-01-04"],
    returns=[0.01, -0.02, 0.03],
)
```

Then check and evaluate the log:

```bash
overfitlab trials validate results/trial_log.csv
overfitlab trials report results/trial_log.csv
```

The format is in the [trial-log contract](trial-log.md), and [trial counting](theory/trial-counting.md) explains what counts as a trial.

## Reproduce the study

```bash
uv sync
uv run olab-study all
```

Real-data steps need the raw files first, and are skipped with a message if they're missing. See `README.md` for the data commands.
