# overfitlab

Statistics that separate a real trading signal from a lucky one.

Every systematic researcher runs many backtests and reports the best. The best of many noisy trials has a high Sharpe ratio even when no trial has any edge. `overfitlab` measures how much of a result that explains.

| Method | Question it answers |
|---|---|
| PSR, MinTRL | Is this Sharpe ratio real, given its length and return shape? |
| DSR, MinBTL | Is it real, given how many trials produced it? |
| PBO via CSCV | Does in-sample selection carry over out of sample? |
| Reality Check, SPA, StepM | Does anything in this set beat the benchmark, and which? |
| Harvey–Liu haircut | What is the Sharpe ratio worth after multiple testing? |
| Purged k-fold, CPCV | How do I cross-validate overlapping labels without leaking? |

## Where to go

- [Getting started](getting-started.md): install it and run a first check.
- [Calculator](calculator.md): deflate a Sharpe ratio in the browser.
- [API reference](api.md).
- [Trial-log contract](trial-log.md): the format the owner's other projects emit.
- [Validation](validation.md): how each method was checked, and what fell short.
- Theory: [PSR and DSR](theory/psr-dsr.md), [PBO](theory/pbo.md), [cross-validation](theory/cv.md), [snooping tests](theory/snooping.md), [haircut](theory/haircut.md), [reference power](theory/power.md), [trial counting](theory/trial-counting.md).

An interactive version of the power curves is [here](assets/power_curves.html).
