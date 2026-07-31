# Confirmatory Locked Test: Criteo visit

## Why This Stage Exists

The earlier locked test estimated a positive but inconclusive advantage for
`s_learner` at the `5.00%` budget. That
interval was wide because the test partition held only a fifth of a
one-million-row audit, not because the effect was known to be zero. The source
experiment has 13,979,592 randomized rows, so the cheapest way to sharpen the
answer is to spend unused rows on evaluation rather than to re-argue the same
numbers.

Nothing is selected here. The champion, the operating budget, the confidence
level, and the estimator were all locked before this sample was drawn.

## Protocol

- Locked champion: `s_learner` (selected earlier, on development data only).
- Refit sample: `data/processed/criteo_audit_1m.parquet` (1,000,000 rows).
- Confirmatory sample: `data/processed/criteo_confirm_4m.parquet` (4,000,000 rows), opened once.
- Verified overlap between the two samples: **0 rows**, by full-row hash.
- Reference policies: `response_model` (the incumbent) and `random_targeting`
  (the floor that shows how much of any gain is ranking skill).
- Treatment propensity used by AIPW: `0.850017`.
- Primary budget `5.00%`, confidence level
  `95.0%`.

## Policy Value

| policy           | budget_pct | n_targeted | incremental_outcome_rate | standard_error_rate | ci_lower_rate | ci_upper_rate | incremental_outcome | ci_lower     | ci_upper     |
| ---------------- | ---------- | ---------- | ------------------------ | ------------------- | ------------- | ------------- | ------------------- | ------------ | ------------ |
| response_model   | 5.000000   | 200000     | 0.003108                 | 0.000145            | 0.002823      | 0.003393      | 12432.564588        | 11292.906665 | 13572.222511 |
| response_model   | 10.000000  | 400000     | 0.005000                 | 0.000189            | 0.004631      | 0.005370      | 20000.855871        | 18522.492209 | 21479.219533 |
| response_model   | 20.000000  | 800000     | 0.006656                 | 0.000214            | 0.006236      | 0.007076      | 26623.672379        | 24944.716459 | 28302.628299 |
| response_model   | 30.000000  | 1200000    | 0.006953                 | 0.000222            | 0.006519      | 0.007387      | 27813.121109        | 26076.446192 | 29549.796025 |
| random_targeting | 5.000000   | 200000     | 0.000424                 | 0.000052            | 0.000323      | 0.000526      | 1697.842283         | 1293.224202  | 2102.460365  |
| random_targeting | 10.000000  | 400000     | 0.000775                 | 0.000073            | 0.000633      | 0.000917      | 3100.475167         | 2531.155798  | 3669.794536  |
| random_targeting | 20.000000  | 800000     | 0.001395                 | 0.000102            | 0.001195      | 0.001595      | 5579.996349         | 4779.012482  | 6380.980216  |
| random_targeting | 30.000000  | 1200000    | 0.002143                 | 0.000125            | 0.001897      | 0.002389      | 8570.333930         | 7586.502423  | 9554.165437  |
| s_learner        | 5.000000   | 200000     | 0.004573                 | 0.000123            | 0.004332      | 0.004814      | 18293.248176        | 17329.073484 | 19257.422868 |
| s_learner        | 10.000000  | 400000     | 0.005738                 | 0.000170            | 0.005405      | 0.006070      | 22951.307058        | 21621.873184 | 24280.740932 |
| s_learner        | 20.000000  | 800000     | 0.006551                 | 0.000206            | 0.006148      | 0.006953      | 26202.193609        | 24590.704045 | 27813.683172 |
| s_learner        | 30.000000  | 1200000    | 0.006845                 | 0.000216            | 0.006422      | 0.007269      | 27381.608743        | 25687.768897 | 29075.448589 |

![Confirmatory policy value](figures/confirmatory_visit_policy_value.png)

## Paired Contrast Against Response Targeting

| policy           | reference_policy | budget_pct | n_targeted | difference_rate | standard_error_rate | difference    | ci_lower      | ci_upper      |
| ---------------- | ---------------- | ---------- | ---------- | --------------- | ------------------- | ------------- | ------------- | ------------- |
| random_targeting | response_model   | 5.000000   | 200000     | -0.002684       | 0.000147            | -10734.722304 | -11886.037069 | -9583.407540  |
| s_learner        | response_model   | 5.000000   | 200000     | 0.001465        | 0.000129            | 5860.683588   | 4850.495592   | 6870.871585   |
| random_targeting | response_model   | 10.000000  | 400000     | -0.004225       | 0.000184            | -16900.380704 | -18339.958768 | -15460.802640 |
| s_learner        | response_model   | 10.000000  | 400000     | 0.000738        | 0.000112            | 2950.451187   | 2075.531057   | 3825.371316   |
| random_targeting | response_model   | 20.000000  | 800000     | -0.005261       | 0.000194            | -21043.676030 | -22568.286521 | -19519.065539 |
| s_learner        | response_model   | 20.000000  | 800000     | -0.000105       | 0.000076            | -421.478770   | -1019.078068  | 176.120527    |
| random_targeting | response_model   | 30.000000  | 1200000    | -0.004811       | 0.000188            | -19242.787179 | -20712.870519 | -17772.703839 |
| s_learner        | response_model   | 30.000000  | 1200000    | -0.000108       | 0.000061            | -431.512366   | -908.181981   | 45.157249     |

At the pre-specified `5.00%` budget the confirmatory
sample confirmed a positive advantage over response targeting at the pre-specified budget. The estimated difference is
`5860.6836` incremental visit outcomes with a
`95.0%` interval of
`[4850.4956, 6870.8716]`.

## Precision Gained

| stage               | n_evaluated | difference_per_100k | standard_error_per_100k | z_statistic | ci_lower_per_100k | ci_upper_per_100k | excludes_zero | standard_error_reduction |
| ------------------- | ----------- | ------------------- | ----------------------- | ----------- | ----------------- | ----------------- | ------------- | ------------------------ |
| Earlier locked test | 200000      | 190.270928          | 55.334839               | 3.438538    | 81.816636         | 298.725220        | True          | 1.000000                 |
| Confirmatory test   | 4000000     | 146.517090          | 12.885288               | 11.370882   | 121.262390        | 171.771790        | True          | 4.294420                 |

Both stages estimate the same locked policy, so the difference between them is
evaluation precision, not a different effect. Figures are normalized per 100,000
evaluated users because totals scale with the sample and are not comparable; the
z-statistic is scale-free and comparable as printed. Agreement between the two
per-100,000 point estimates is the check that the larger sample bought precision
rather than a new answer.

## Ranking Metrics

| policy           | benchmark_relative_auuc |
| ---------------- | ----------------------- |
| response_model   | 0.009130                |
| s_learner        | 0.009037                |
| random_targeting | 0.004958                |

Relative AUUC scores the whole ranking. The decision is made on the
budget-specific paired contrast because the campaign can only treat a fixed
share of users.

## Runtime

| stage       | model                   | fit_seconds |
| ----------- | ----------------------- | ----------- |
| locked_test | response_model          | 11.771896   |
| locked_test | random_targeting        | 0.000052    |
| locked_test | s_learner               | 9.295238    |
| locked_test | aipw_nuisance_t_learner | 11.592130   |

## Interpretation Boundaries

- The confirmatory sample is disjoint from every earlier sample but is drawn
  from the same source experiment and the same time period.
- The interval conditions on the fitted policies; it does not carry the
  uncertainty of having chosen `s_learner` in the first place. Selection
  stability is measured separately by the repeated-split experiment.
- AIPW treats the treatment propensity as known, as is standard for a
  randomized design; the propensity is estimated from the refit sample.
- Offline evidence bounds what a live campaign might do. It does not replace a
  randomized online test, and it carries no claim about revenue.

## Reproducible Outputs

- Policy values: `outputs/tables/confirmatory_visit_test.csv`
- Paired contrasts: `outputs/tables/confirmatory_visit_contrasts.csv`
