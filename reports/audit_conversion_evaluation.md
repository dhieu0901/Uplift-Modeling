# Honest Uplift Model Selection: Criteo conversion

## Locked Protocol

- Data: `data/processed/criteo_audit_1m.parquet` (1,000,000 rows), outcome `conversion`.
- Base train/validation/test fractions: `0.60` / `0.20` / `0.20`.
- Selection folds: `1` over `200,000` selection observations.
- Primary targeting budget: `5.00%`.
- Confidence level: `95.0%`.
- Candidate model and hyperparameter selection uses one explicit validation holdout; locked-test outcomes are excluded.
- The champion is the candidate with the largest paired AIPW lower confidence bound against response targeting at the primary budget.
- After selection, the champion and response baseline are refit on train + validation.
- Locked-test outcomes are opened once for the final comparison.

## Out-of-Sample Development Selection

| policy               | budget_pct | n_targeted | incremental_outcome_rate | standard_error_rate | ci_lower_rate | ci_upper_rate | incremental_outcome | ci_lower   | ci_upper   | benchmark_relative_auuc | difference_vs_response | difference_ci_lower | difference_ci_upper |
| -------------------- | ---------- | ---------- | ------------------------ | ------------------- | ------------- | ------------- | ------------------- | ---------- | ---------- | ----------------------- | ---------------------- | ------------------- | ------------------- |
| undersampled_t_lr_k5 | 5.000000   | 10000      | 0.000804                 | 0.000236            | 0.000342      | 0.001266      | 160.890755          | 68.490119  | 253.291390 | 0.000958                | -40.706717             | -98.062101          | 16.648667           |
| response_model       | 5.000000   | 10000      | 0.001008                 | 0.000259            | 0.000501      | 0.001515      | 201.597472          | 100.184450 | 303.010494 | 0.001179                | nan                    | nan                 | nan                 |

Selected champion: **undersampled_t_lr_k5**.

## Locked-Test Policy Value

| policy               | budget_pct | n_targeted | incremental_outcome_rate | standard_error_rate | ci_lower_rate | ci_upper_rate | incremental_outcome | ci_lower   | ci_upper   |
| -------------------- | ---------- | ---------- | ------------------------ | ------------------- | ------------- | ------------- | ------------------- | ---------- | ---------- |
| response_model       | 5.000000   | 10000      | 0.000940                 | 0.000262            | 0.000426      | 0.001454      | 188.092769          | 85.295016  | 290.890523 |
| response_model       | 10.000000  | 20000      | 0.001062                 | 0.000276            | 0.000521      | 0.001603      | 212.363723          | 104.105999 | 320.621447 |
| response_model       | 20.000000  | 40000      | 0.001099                 | 0.000290            | 0.000530      | 0.001668      | 219.780211          | 105.941956 | 333.618466 |
| response_model       | 30.000000  | 60000      | 0.001163                 | 0.000291            | 0.000592      | 0.001734      | 232.616508          | 118.377128 | 346.855888 |
| undersampled_t_lr_k5 | 5.000000   | 10000      | 0.000705                 | 0.000241            | 0.000232      | 0.001178      | 140.951339          | 46.368537  | 235.534141 |
| undersampled_t_lr_k5 | 10.000000  | 20000      | 0.000773                 | 0.000258            | 0.000269      | 0.001278      | 154.661445          | 53.710510  | 255.612381 |
| undersampled_t_lr_k5 | 20.000000  | 40000      | 0.000808                 | 0.000271            | 0.000276      | 0.001339      | 161.544573          | 55.297377  | 267.791770 |
| undersampled_t_lr_k5 | 30.000000  | 60000      | 0.000837                 | 0.000276            | 0.000295      | 0.001378      | 167.375677          | 59.058886  | 275.692469 |

![Locked-test policy value](figures/audit_conversion_policy_value.png)

## Paired Contrast Against Response Targeting

| policy               | reference_policy | budget_pct | n_targeted | difference_rate | standard_error_rate | difference | ci_lower    | ci_upper   |
| -------------------- | ---------------- | ---------- | ---------- | --------------- | ------------------- | ---------- | ----------- | ---------- |
| undersampled_t_lr_k5 | response_model   | 5.000000   | 10000      | -0.000236       | 0.000147            | -47.141430 | -104.619494 | 10.336634  |
| undersampled_t_lr_k5 | response_model   | 10.000000  | 20000      | -0.000289       | 0.000123            | -57.702278 | -106.050402 | -9.354154  |
| undersampled_t_lr_k5 | response_model   | 20.000000  | 40000      | -0.000291       | 0.000108            | -58.235637 | -100.582569 | -15.888706 |
| undersampled_t_lr_k5 | response_model   | 30.000000  | 60000      | -0.000326       | 0.000106            | -65.240830 | -106.822781 | -23.658880 |

At the pre-specified `5.00%` budget, the locked test
was inconclusive relative to response targeting. The estimated difference is `-47.1414`
incremental conversion outcomes with a confidence interval of
`[-104.6195, 10.3366]`.

## Ranking Metrics

| policy               | benchmark_relative_auuc |
| -------------------- | ----------------------- |
| response_model       | 0.001173                |
| undersampled_t_lr_k5 | 0.000961                |

AUUC is secondary. The decision is based on the budget-specific AIPW policy
contrast because the campaign has a fixed operating budget.

## Runtime

| stage       | model                   | fit_seconds |
| ----------- | ----------------------- | ----------- |
| selection   | response_model          | 21.305238   |
| selection   | undersampled_t_lr_k5    | 0.562037    |
| selection   | aipw_nuisance_t_learner | 22.307650   |
| locked_test | response_model          | 21.380687   |
| locked_test | undersampled_t_lr_k5    | 0.512429    |
| locked_test | aipw_nuisance_t_learner | 23.394973   |

## Statistical Scope

The AIPW intervals account for evaluation-sample uncertainty conditional on the
locked fitted policies. Repeated honest splits are required to quantify training
and selection instability. No production-impact claim is made without a live
randomized experiment.

## Reproducible Outputs

- Selection values: `reports/tables/audit_conversion_selection.csv`
- Locked-test values: `reports/tables/audit_conversion_test.csv`
- Paired contrasts: `reports/tables/audit_conversion_contrasts.csv`
