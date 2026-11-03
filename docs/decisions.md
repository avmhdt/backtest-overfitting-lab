# Decisions

Copied at M1 from the owner's `DECISIONS.md` (recorded 2026-10-31), which overrides the spec. Agent decisions on ambiguous points follow. API keys never go in this file.

## Owner decisions that apply here

- **Project:** #2 `backtest-overfitting-lab`, in the first batch with #3 `crypto-lob-alpha` and #4 `vol-surface-modeling`. Scope is all 15 projects, for researcher and developer seats, a few at a time, with no fixed date.
- **Merging:** agents push feature branches and open PRs. PRs merge automatically once CI is green. Branch protection and auto-merge get turned on per repo once its CI exists (M1).
- **GitHub:** the `gh` active account is `avmhdt`, with admin on the repo. Actions is enabled.
- **WRDS:** no access. CRSP, Compustat, OptionMetrics and Capital IQ paths stay off.
- **Sharadar:** Full History Bundle bought directly from Sharadar, not through Nasdaq Data Link.
  - Base URL `https://api.sharadar.com/v1.0`, key in `SHARADAR_API_KEY`, sent in the `x-api-key` header so it stays out of URLs and logs.
  - Tables `stocks` (SEP), `funds` (SFP), `fundamentals` (SF1), `tickers`, `actions`. Bulk zips with `years=full`.
  - This project's spec was rewritten on that basis.
- **Licence:** Personal Use License. Sharadar replied in writing on 2026-10-31 that public code, aggregate results, job applications, aggregate cross-checks and hosted coding tools are allowed. Publishing stays aggregate-only, with no security-level data.
- **Money:** Sharadar is an existing subscription. Nothing else is approved for this project (USD 0).
- **Storage:** stay on `D:`. Free space is checked at each milestone. Project data never goes on `C:` or into git.
- **Waiting on the owner:** put the Sharadar key in `.env`.
- **Accepted:** every other owner decision in section 3 of the spec with a default stands as written. None of the changes from spec defaults in `DECISIONS.md` apply to this project.

## Spec decisions

| ID | Decision | Status |
|---|---|---|
| D-01 | Package and import name `overfitlab` | Accepted. Checked 2026-11-01: `pypi.org/simple/overfitlab/`, `test.pypi.org/simple/overfitlab/` and both JSON APIs return 404. The HTML project pages return 200 to scripts because they serve a bot-challenge shell, so the simple index is the reliable check. Re-check before the first release. |
| D-02 | Publish to PyPI | Default: (a) TestPyPI then PyPI on `v0.1.0` once P-12 to P-14 are done, otherwise (c). |
| D-03 | Licence | MIT, code only. Licensed data is never redistributed. |
| D-04 | Paid data spend | No new spend. |
| D-05 | Universe | Default stands. |
| D-06 | Verdict thresholds | Default stands: `dsr_min = 0.95`, `pbo_max = 0.20`, `snoop_alpha = 0.05`, BHY haircut Sharpe above 0. |
| D-07 | Monte Carlo budget | Default stands. |
| D-08 | Trials in the headline DSR | Default stands: raw N in the headline, clustered N alongside. |
| D-09 | Kurtosis convention | Non-excess (normal = 3) everywhere in the public API. |
| D-10 | Annualisation | 252, 52 and 12. Golden tests pass the paper's own value. A log's `x_periods_per_year` overrides the inferred value. |
| D-11 | Contract freeze | Default stands: freeze v1.0 at the end of M2. |
| D-12 | Docs host | Default stands: GitHub Pages. |
| D-13 | Python versions | 3.12 and 3.13 in CI, `requires-python = ">=3.12"`. |

## Owner rulings, 2026-11-01

- **AC-W-07 stays open.** The owner asked for a stricter check. The raw-against-raw comparison with `arch` is kept as evidence and does not close the criterion. The limit is unchanged. An independent reference that really studentises is needed; the candidates are in [validation](validation.md).
- **G-03c is a documented exception to AC-W-03.** The tolerance is unchanged and the case still reports `pass=false`. It misses by 5e-6 because the paper feeds an input printed to three decimals, so the exception covers this one case and no other. The AC-W-03 row of the local spec records it.
- **G-07 is sourced** (2026-11-02): *Advances in Financial Machine Learning*, chapter 12, section 12.4.1, p. 164, with Figures 12.1 and 12.2, checked in the owner's copy. Only the two counts and the citation are recorded.
- **Deviation from P-02:** the repo's default workflow permission stays at `read`, not read/write as P-02 asks. Workflows declare their own `permissions:`, so M1 to M4 do not need the default raised.

- **AC-W-02: the DSR miss is kept as a finding** (2026-11-02). The `dsr` rows of `power_check.csv` keep reporting the miss, and CI keeps the strict expected failure. The fixed-V diagnostic is post hoc and does not count toward the criterion. M5 proceeds on that basis, and the writeup must explain the cause.
- **AC-W-07: Octave and R are approved** (2026-11-02). `bsds.m` under GNU Octave is the studentised reference for SPA and the Reality Check, and R's `StepwiseTest` for StepM. Case 2's miss is documented with its cause, the limit does not change, and no paid licence is approved.

## Agent decisions

- **M1, `DATA_ROOT` on Windows:** `.env.example` is fixed by the spec and sets `DATA_ROOT` to a `~` path meant for WSL. On native Windows the settings loader ignores a `DATA_ROOT` that starts with `~`, so a copied `.env` cannot put project data on `C:`.
- **M1, study in the dev group:** `overfitlab-study` is a dev dependency of the root project, so `uv sync` followed by `uv run olab-study` works from the repo root. The published wheel still contains only `overfitlab`.
- **M1, line endings:** `.gitattributes` forces LF, because the trial-log dialect and `ruff format` both need LF on Windows too.
- **M1, `.gitignore`:** besides `results/logs/`, added `results/sims/` and tool caches, plus a negation for `study/src/overfitlab_study/data/`, because the existing `data/` rule also matched that code package. No existing rule was removed.
- **M1, action pins:** `astral-sh/setup-uv` publishes no moving major tag, so `ci.yml` pins an exact release.
- **M2, flat series:** `sharpe_ratio` returns 0.0 for a series with zero variance, so a flat factory variant does not abort a matrix evaluation. `return_moments` still raises, because skew and kurtosis are undefined there.
- **M2, skewness and kurtosis:** moment ratios without a small-sample correction, matching `scipy.stats.skew` and `kurtosis(fisher=False)` defaults, which is what the PSR code in Bailey and López de Prado's appendix expects.
- **M2, haircut conventions:** follows `Haircut_SR.m`, not the paper's Exhibit 7, where they differ; see [validation](validation.md). `haircut_sharpe` returns the autocorrelation-corrected Sharpe ratio in `sr_annual`, and `haircut_pct` is relative to it, as in Exhibit 5.
- **M2, simulated golden tolerances:** the spec says Holm and BHY match "within the printed precision because they are simulated". The authors' values come from 2,000 unseeded runs, so half a unit of the third decimal is below their own noise. Tolerances are 3 standard deviations of a 2,000-run median, measured before any comparison with the printed values.
- **M2, G-03c:** a known miss of 5e-6 past tolerance. The tolerance was left alone; see [validation](validation.md) and the owner's ruling above.
- **M2, contract details settled before the freeze (D-11):**
  - `E_SHARPE` was added, because the spec's error list had no code for an unreadable `sharpe_annual`.
  - `W_SUMMARY_ONLY` was added. The Parquet files are git-ignored, so `overfitlab trials validate results/trial_log.csv` in a sibling's CI sees none. Such a log passes on its CSV checks with this warning. If any series exists, a missing one is `E_RETURNS_MISSING`.
  - `--no-strict` and `strict=False` list errors without failing. `--strict` is the default.
  - Rows with `n_obs = 0` may leave `start` and `end` empty. An empty `x_periods_per_year` cell means absent.
  - `E_FREQ_MIX` applies per family in the file validator, and across the whole scope in the evaluator, so families at different frequencies may share a log.
  - `TrialLogger` writes `x_periods_per_year` and `x_note` in every new log. An existing log keeps its header, since the file is append-only.
  - The `-dirty` check ignores the log file itself, otherwise every run after the first would be marked dirty.
- **M2, `evaluate_trial_log`:** ships in M2 with N, V, SR₀, DSR and the haircut. PBO, the snooping tests and the verdict are added in M4, and until then the verdict is `INCONCLUSIVE` with a reason. `trials report` and `trials add-manual` also ship in M2, because the frozen contract refers to them.
- **M3, embargo:** the window of `ceil(embargo_pct * T)` rows starts after the last row purged behind a test fold, not right after the fold. Right after the fold it would sit inside the purge whenever labels span several days and remove nothing. With point labels the two readings coincide.
- **M3, `t0`:** the default of row positions works only with integer `t1`, so datetime `t1` needs an explicit `t0`.
- **M3, factory:** breakout and mean-reversion run a long-short state machine and long-only clips it at zero, the same rule as the moving-average family. With `exit_z = 0.0` the spec's rule `|z| < exit_z` never fires, so those variants hold until the opposite entry; this is kept as written. Rolling standard deviations use `ddof=1`. `exit_length` uses Python's `round`.
- **M3, `run_strategies`:** `T + 1` closes give a `T x N` matrix, because the first close has no return.
- **M3, `planted_signal_prices`:** the trend oracle is `ma_cross` 10/60 and the mean-reversion oracle is `mean_reversion` 20, 1.0, 0.0, both long-short. The amplitude is calibrated by bisection on one seeded 100,000-bar path, before costs.
- **M3, `bootstrap.py`:** arrives in M3 instead of M4, because `block_bootstrap_returns` needs it.
- **M3, generators:** `gaussian_noise` and `student_t_noise` always return `T x N`, including when `N = 1`. `gbm_prices` returns `n_obs` prices, so a run of `T` returns needs `T + 1` prices.
- **M3, G-07:** see the owner's rulings above. `tests/unit/test_cv.py` also asserts 15 splits and 5 paths against the formula in FR-09.
- **M4, `arch` comparison:** `arch` 8 does not studentise, so the 0.02 and 1e-8 checks compare the Reality Check and `spa_test(studentize=False)` with `arch`. This is evidence and does not close AC-W-07; see the owner's ruling above. The studentised gap is reported as `max_studentised_gap`. `studentize` is an added keyword on `spa_test`, and its default is Hansen's statistic.
- **M4, PBO:** rows beyond the last whole block are dropped, as in the paper. A logit of exactly zero counts toward PBO, as the spec's `λ ≤ 0` says.
- **M4, block length:** the default is the median Politis-White length over the columns. The gate uses one set of resamples for SPA and the stepdown.
- **M4, `evaluate` needs at least nine observations for the snooping tests:** with fewer, `evaluate_trial_log` reports them as not computed.
- **M4, `TrialLogReport`:** gains `spa_pvalue` and `stepm_rejects_selected`.
- **M4, clustered N:** average linkage, the number of clusters is scanned on up to 40 geometric steps, and a best mean silhouette below 0.05 counts as no structure, so every trial stays independent.
- **M4, noise table:** `noise_experiment.csv` holds one summary row per setting. The per-seed values stay in the git-ignored `results/sims/noise/`.
- **M5, AC-W-02:** the three DSR rows with N > 1 miss the reference, which the owner kept as a finding; see the rulings above. `dsr` estimates V from the same trials, and the reference assumes V is known. `dsr_known_v` fixes V at 1/(T − 1). It is post hoc: it was added after the failure, so it does not count.
- **M5, power tests:** a test detects when it rejects on the best trial, whichever trial that is, because the reference power is the probability that the maximum clears the bar. PSR is the naive PSR of the best trial. The haircut rejects when the BHY adjusted p-value is at most α, with 1,000 simulated sets per replication. PBO "rejects" at most at 0.20, the D-06 limit.
- **M5, power bootstrap:** block length 1, because the simulated returns are IID. The closed-form tests run 1,000 replications and the rest 500, per D-07. The PBO trend uses the SR grid at 5 years and N = 100.
- **M5, `^spx`:** not in the Stooq snapshot, so `^ndx` is the nearest available index (D-05).
- **M5, ETF history:** Stooq's U.S. ETF files start in 2005, and Sharadar's reach further back. The case study uses the dates all 12 series share, from 2006.
- **M5, Sharadar:** `close` in `prices_daily` is `closeadj`, and the other prices are scaled by the same ratio, so returns include distributions. The key is sent in `x-api-key`, as the owner decided, though Sharadar's docs show a query parameter.
- **M5, `STOOQ_DIR`:** the owner's `.env` does not set it, so `data stooq` takes `--stooq-dir`.
- **M5, factory case:** a trial is one variant across the whole universe, equal weighted. One trial per symbol and variant would give 12,408 rows, over the 1 MB limit on tracked files, and would multiply the count without adding independent ideas. The first 250 bars are dropped, so every variant is live throughout. ETFs and the index are excess of the French risk-free rate. FX is already excess (DQ-07).
- **M5, holdout:** `x_is_holdout = true` in a separate family, because its dates do not overlap the in-sample trials. A rerun reads the holdout back from the log and never recomputes it.
- **M5, factors case:** `F-F_Research_Data_Factors` plus the four AQR datasets. `RF` is excluded. Columns needing a window shorter than the common one are reported as dropped.
- **M5, OSAP:** the owner supplied release 2026.10 of the OP set, so the `openassetpricing` package was not needed and nothing was downloaded twice. `cases.osap` in `study.yaml` now reads `release: "2026.10", weighting: op`, because that set uses each original paper's weighting, not value weighting throughout. Value weighting is not needed: the question is whether the published results survive, and the OP set is the published construction.
- **M5, OSAP windows:** in-sample is the paper's sample years, and post-publication is after the publication year. For StepM a predictor's months with no portfolio are 0.0, which leaves its t-statistic unchanged. Predictors need 60 months in a window.
- **M5, `git_sha`:** a study run records the commit it started from, and `results/` and `reports/` do not mark it dirty.
- **`validate studentised`:** a command added to section 10 of the spec for the owner's AC-W-07 ruling. It runs locally because it needs Octave and R, and CI keeps the `arch` comparison. `bsds.m` is pinned to commit `9622b6c546bc` and kept out of the repo, because the toolbox has no licence file. `StepwiseTest` installs into the WSL user library through `scripts/r/install_stepwise.R`.
- **`.env`:** `STOOQ_DIR` was added on 2026-11-02, as the owner approved, so `data stooq` no longer needs `--stooq-dir`. The key was not read.
- **M6, CI:** the coverage job took 44 minutes because it ran the `xcheck` tests through pure-Python loops. It now leaves them to `study-quick`, which runs them with Numba on.
- **M6, `all`:** runs the study commands on the tables already in `data/`. Ingest stays separate, because Sharadar and French need the network, and a missing table skips the case studies with a message. `--quick` also skips `validate arch`, which CI runs at full size, and `bench`, because timings differ on every run. `validate studentised` is not part of `all`, because it needs Octave and R.

- **M6, docs:** theory pages quote the committed tables, and `tests/unit/test_docs.py` checks the API page against `__all__`, the docstring sections, the word limit and the getting-started examples. `import overfitlab` is tested not to load pyarrow, typer or Polars, and its time stays in `olab-study bench`.
- **M6, haircut:** `haircut_sharpe` returned `inf` once the adjusted p-value fell below about 1e-16, because `1 - p/2` rounded to 1. It now inverts the lower tail and caps the result at the t-statistic. One row of `case_factors` (AQR's Intl Stock Selection Multi-style, t = 9.8) had shown `inf`.
- **M6, calculator:** `npm test` runs `node --test "js/test/*.test.mjs"`, because Node 22 does not accept the spec's directory form, `node --test js/test/`. MinTRL is against a benchmark of 0, MinBTL takes the entered Sharpe ratio as the target, and autocorrelation enters only the haircut, as in the CLI.
- **M6, FR-34:** the Holm and BHY lookup is a Could and is deferred. The page gives the Bonferroni haircut, which is exact, and points to the library for the other two.
- **M6, release:** `release.yml` also runs by hand, to prove Trusted Publishing and the install without a tag. A manual run builds `<version>.dev<run>`, so the real version stays free on TestPyPI, and the PyPI job runs only on a pushed `v*` tag. The tag must match `pyproject.toml`. `scripts/smoke_install.py` runs the README quickstart from a fresh venv.
- **M6, full run:** the AC-W-09 run started from `1e40df2`. Every table matched the commit except benchmark times, `case_factors` (the haircut fix) and `own_projects` (below). Rows from `sim leakage` on carry `-dirty` because `sim power` rewrote `docs/assets/power_curves.html` with its first pinned id, and nothing else in the tree had changed. The rows at `8021ed2` and `2760f4f` are a dry run made while the docs were being edited.
- **M6, own projects:** `equity-factor-pipeline` now logs 517 trials in eight families, up from 20 in one. FR-29 evaluates each family alone, and three pass. That is a statement about each family's own search: with N = 517 the bar for `investment` rises from 0.70 to 1.02, above its Sharpe ratio of 1.00.
- **M7, own projects:** the `equity-factor-pipeline` log moved from 517 to 613 trials between runs while it was being rebuilt, so the table and the prose that quotes it are provisional until that rebuild lands. `own_projects` now also reports each family's DSR at the project's whole trial count (`n_raw_project`, `sr0_annual_project_n`, `dsr_project_n`). These columns add to FR-29 and do not change its verdicts.
- **M7, README:** rewritten as the research memo. The checklists stay, with their ids, under the heading "Depth" in place of "Impresses hirers", because the README is also the PyPI page. `tests/unit/test_docs.py` fails if the README or memo stops matching the committed tables.
- **M7, final run:** the M7 exit run started from `e7566b6`, after the reboot. Every table and figure matched the commit except benchmark times. The `equity-factor-pipeline` log still had 613 trials, so the README and memo numbers stand. Rows from `sim leakage` on carry `-dirty` because Plotly wrote `power_curves.html` with Windows line endings. The content was identical, and the chart is now written with `\n` endings.
- **M7, own projects final:** the owner confirmed on 2026-11-03 that the `equity-factor-pipeline` numbers are final. Its PR 9 rebuilt the project from a fresh clone and added no trials, and regenerating `own_projects` from its merged log gives the committed table byte for byte. The README and memo no longer mark the table provisional.
- **M7, AQR permission:** AQR's terms of use bar publishing information from its site without written consent. On 2026-11-03 AQR gave written permission to publish aggregate statistics and charts with citation, with no redistribution of its files or raw series. The owner holds the email. AQR is cited wherever its results appear, and its files stay in the git-ignored `data/raw/aqr/`.
- **M7, superseded French-only run:** before the permission arrived, PR #17 withheld AQR results and reran the factors case on French data alone. It was closed without merging. That run's experiment-log row (`c6fbb18a`, commit `daa6f39` on branch `aqr-withheld`) is kept, because the log is append-only, and is superseded by the permission. Its results are not published.
- **v0.1.0, AC-H-03:** `v0.1.0` was tagged on `c612356` and published to PyPI on 2026-11-03, with a wheel and an sdist, and is listed in `CHANGELOG.md` after the owner approved the `pypi` environment. `pip install overfitlab==0.1.0` from PyPI into a fresh venv then passed `scripts/smoke_install.py`, which runs the README quickstart, the CLI and the light-import check. It passed on Windows 11 with Python 3.12.10, and on Ubuntu 22.04 under WSL with Python 3.12.15 from `uv venv --seed`, because the system Python is 3.10. AC-H-03 is met.
