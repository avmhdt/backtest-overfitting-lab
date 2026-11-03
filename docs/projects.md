# Trial logs in practice

`equity-factor-pipeline` is the first of the owner's projects to emit a trial log in the [contract's](trial-log.md) format. `olab-study projects` evaluates each strategy family in it, and the full table is `reports/tables/own_projects.csv`.

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

The [memo](memo.md) discusses what this means for judging research.
