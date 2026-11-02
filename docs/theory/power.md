# Reference power

The power check compares each test's simulated rejection rate with a closed form. The closed forms live in `study/src/overfitlab_study/sims/theory.py` and share no code with the tests.

**Setup.** N strategies have T independent, normal returns. Exactly one has a true per-period Sharpe ratio SR, and the others have none. Then δ = SR·√T is the mean of the planted strategy's score, and the other scores are standard normal.

## PSR with one trial

The estimator has variance about (1 + SR²/2)/(T − 1) for normal returns. PSR(0) rejects when the estimated Sharpe ratio, divided by its standard error under the null variance, exceeds z₁₋α. So:

power = Φ( SR·√(T − 1) / √(1 + SR²/2) − z₁₋α )

## One planted strategy among N

Every max-based test rejects when the largest of the N scores exceeds some critical value c. The maximum stays below c only if all the scores do, so:

power = 1 − Φ(c − δ)·Φ(c)^(N−1)

The tests differ only in c.

**Reality Check, SPA and StepM.** The bootstrap critical value is the 1 − α quantile of the maximum of N null scores. Independence gives Φ(c)^N = 1 − α, so c = Φ⁻¹((1 − α)^(1/N)).

**DSR.** It rejects when the best Sharpe ratio beats the expected maximum SR₀ by z₁₋α standard errors. In score units that is c = e_N + z₁₋α, with e_N the expected maximum of N standard normals, from the approximation of FR-05.

## What the formulas leave out

- The DSR reference treats V as known, at the null value 1/T. The DSR estimates V from the same trials, and the two differ a lot. A large best Sharpe ratio also inflates the sample variance, so the bar rises exactly when the maximum is high, which costs power, most of all at small N. `power_check.csv` reports both: `dsr` as used in practice, and a `dsr_known_v` diagnostic that fixes V at 1/(T − 1).
- With N > 1, a rejection can come from a noise strategy. That is what these tests measure: whether anything in the set is real, not which one.
- The haircut and PBO have no closed reference here. PBO is checked for its trend: the median must fall as the true Sharpe ratio rises.
