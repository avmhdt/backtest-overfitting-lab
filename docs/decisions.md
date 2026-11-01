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

## Agent decisions

- **M1, `DATA_ROOT` on Windows:** `.env.example` is fixed by the spec and sets `DATA_ROOT` to a `~` path meant for WSL. On native Windows the settings loader ignores a `DATA_ROOT` that starts with `~`, so a copied `.env` cannot put project data on `C:`.
- **M1, study in the dev group:** `overfitlab-study` is a dev dependency of the root project, so `uv sync` followed by `uv run olab-study` works from the repo root. The published wheel still contains only `overfitlab`.
- **M1, line endings:** `.gitattributes` forces LF, because the trial-log dialect and `ruff format` both need LF on Windows too.
- **M1, `.gitignore`:** besides `results/logs/`, added `results/sims/` and tool caches, plus a negation for `study/src/overfitlab_study/data/`, because the existing `data/` rule also matched that code package. No existing rule was removed.
- **M1, action pins:** `astral-sh/setup-uv` publishes no moving major tag, so `ci.yml` pins an exact release.
