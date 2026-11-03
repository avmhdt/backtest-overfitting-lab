# Changelog

All notable changes to `overfitlab` are recorded here. Versions follow [semantic versioning](https://semver.org/).

## 0.1.1 (2026-11-03)

README only, so the PyPI page reads as a package description. The code is unchanged.

### Changed

- The title names the package.
- The data table states that Sharadar is a paid subscription and no longer breaks around the AQR note.
- The per-family trial-log table moved to the docs site's [trial logs in practice](https://avmhdt.github.io/backtest-overfitting-lab/projects/) page.
- The project checklists moved to the docs site's [project criteria](https://avmhdt.github.io/backtest-overfitting-lab/criteria/) page.
- "What failed" describes each item in plain words, without internal ids.
- "How to reproduce" says it needs the GitHub repository, because `olab-study` isn't in the PyPI package.

## 0.1.0 (2026-11-03)

First release, published to [PyPI](https://pypi.org/project/overfitlab/0.1.0/) from tag `v0.1.0` on commit `c612356`.

### Added

- Sharpe statistics: `sharpe_ratio`, `probabilistic_sharpe_ratio` and `min_track_record_length`, with skew and non-excess kurtosis, and Lo's AR(1) annualisation.
- Deflated Sharpe Ratio: `deflated_sharpe_ratio`, `expected_max_sharpe` and `min_backtest_length`.
- Harvey–Liu haircut: `haircut_sharpe`, with Bonferroni, Holm and BHY.
- Cross-validation: `PurgedKFold` and `CombinatorialPurgedCV` with purging and embargo, and `assemble_paths`.
- Probability of Backtest Overfitting by CSCV: `pbo_cscv`.
- Data-snooping tests: `reality_check`, `spa_test` and `romano_wolf_stepdown`, on a stationary bootstrap.
- Synthetic data with known truth, and a strategy factory of crossover, breakout and mean-reversion variants.
- Verdict gate: `evaluate`, which requires the DSR, PBO, SPA, StepM and the haircut to pass together.
- Trial-log contract v1.0: `TrialLogger`, `read_trial_log`, `validate_trial_log` and `evaluate_trial_log`.
- The `overfitlab` command line: `psr`, `dsr`, `mintrl`, `minbtl`, `haircut`, `pbo`, `snoop`, and `trials validate`, `trials report` and `trials add-manual`.
- Docs site with theory pages, the API reference, a Sharpe-haircut calculator and the research memo.
