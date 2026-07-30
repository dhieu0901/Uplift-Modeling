# Honest Uplift Model Selection: Criteo conversion

## Locked Protocol

- Data: `data/processed/criteo_audit_1m.parquet` (1,000,000 rows), outcome `conversion`.
- Base train/validation/test fractions: `0.60` / `0.20` / `0.20`.
- Selection folds: `3` over
  `800,000` selection observations.
- Primary targeting budget: `5.00%`.
- Confidence level: `95.0%`.
- Candidate model and hyperparameter selection uses 3-fold out-of-fold predictions on the combined development sample;
  locked-test outcomes are excluded.
- The champion is the candidate with the largest paired AIPW lower confidence
  bound against response targeting at the primary budget.
- Reference policies `response_model` and `random_targeting` are always
  evaluated but can never win selection.
- After selection, the champion and reference policies are refit on
  train + validation.
- Locked-test outcomes are opened once for the final comparison.

## Out-of-Sample Development Selection

| policy               | budget_pct | n_targeted | incremental_outcome_rate | standard_error_rate | ci_lower_rate | ci_upper_rate | incremental_outcome | ci_lower   | ci_upper   | benchmark_relative_auuc | difference_vs_response | difference_ci_lower | difference_ci_upper |
| -------------------- | ---------- | ---------- | ------------------------ | ------------------- | ------------- | ------------- | ------------------- | ---------- | ---------- | ----------------------- | ---------------------- | ------------------- | ------------------- |
| undersampled_t_lr_k1 | 5.000000   | 40000      | 0.000788                 | 0.000120            | 0.000552      | 0.001023      | 630.338421          | 441.934021 | 818.742821 | 0.001042                | -43.835116             | -143.071049         | 55.400818           |
| random_targeting     | 5.000000   | 40000      | 0.000044                 | 0.000032            | -0.000018     | 0.000106      | 35.249942           | -14.630071 | 85.129956  | 0.000700                | -638.923595            | -836.234510         | -441.612680         |
| response_model       | 5.000000   | 40000      | 0.000843                 | 0.000129            | 0.000590      | 0.001095      | 674.173537          | 472.323142 | 876.023931 | 0.001156                | nan                    | nan                 | nan                 |

Selected champion: **undersampled_t_lr_k1**.

## Locked-Test Policy Value

| policy               | budget_pct | n_targeted | incremental_outcome_rate | standard_error_rate | ci_lower_rate | ci_upper_rate | incremental_outcome | ci_lower   | ci_upper   |
| -------------------- | ---------- | ---------- | ------------------------ | ------------------- | ------------- | ------------- | ------------------- | ---------- | ---------- |
| response_model       | 5.000000   | 10000      | 0.000997                 | 0.000261            | 0.000484      | 0.001509      | 199.307735          | 96.888582  | 301.726889 |
| response_model       | 10.000000  | 20000      | 0.001168                 | 0.000273            | 0.000633      | 0.001703      | 233.576949          | 126.623530 | 340.530368 |
| response_model       | 20.000000  | 40000      | 0.001105                 | 0.000290            | 0.000535      | 0.001674      | 220.938552          | 107.076085 | 334.801019 |
| response_model       | 30.000000  | 60000      | 0.001160                 | 0.000292            | 0.000589      | 0.001732      | 232.021163          | 117.725308 | 346.317017 |
| random_targeting     | 5.000000   | 10000      | 0.000018                 | 0.000064            | -0.000106     | 0.000143      | 3.638836            | -21.297189 | 28.574860  |
| random_targeting     | 10.000000  | 20000      | 0.000008                 | 0.000106            | -0.000200     | 0.000216      | 1.613860            | -39.937161 | 43.164881  |
| random_targeting     | 20.000000  | 40000      | 0.000211                 | 0.000127            | -0.000038     | 0.000460      | 42.177272           | -7.594104  | 91.948648  |
| random_targeting     | 30.000000  | 60000      | 0.000370                 | 0.000166            | 0.000044      | 0.000697      | 74.095790           | 8.839684   | 139.351896 |
| undersampled_t_lr_k1 | 5.000000   | 10000      | 0.000709                 | 0.000252            | 0.000214      | 0.001203      | 141.738970          | 42.899351  | 240.578589 |
| undersampled_t_lr_k1 | 10.000000  | 20000      | 0.000918                 | 0.000263            | 0.000403      | 0.001432      | 183.505472          | 80.552670  | 286.458274 |
| undersampled_t_lr_k1 | 20.000000  | 40000      | 0.000987                 | 0.000273            | 0.000451      | 0.001522      | 197.312328          | 90.145038  | 304.479617 |
| undersampled_t_lr_k1 | 30.000000  | 60000      | 0.000965                 | 0.000283            | 0.000411      | 0.001518      | 192.926171          | 82.160961  | 303.691381 |

![Locked-test policy value](figures/audit_conversion_policy_value.png)

## Paired Contrast Against Response Targeting

| policy               | reference_policy | budget_pct | n_targeted | difference_rate | standard_error_rate | difference  | ci_lower    | ci_upper    |
| -------------------- | ---------------- | ---------- | ---------- | --------------- | ------------------- | ----------- | ----------- | ----------- |
| random_targeting     | response_model   | 5.000000   | 10000      | -0.000978       | 0.000254            | -195.668899 | -295.054305 | -96.283494  |
| undersampled_t_lr_k1 | response_model   | 5.000000   | 10000      | -0.000288       | 0.000121            | -57.568765  | -105.066431 | -10.071099  |
| random_targeting     | response_model   | 10.000000  | 20000      | -0.001160       | 0.000252            | -231.963089 | -330.684428 | -133.241750 |
| undersampled_t_lr_k1 | response_model   | 10.000000  | 20000      | -0.000250       | 0.000102            | -50.071477  | -90.240570  | -9.902385   |
| random_targeting     | response_model   | 20.000000  | 40000      | -0.000894       | 0.000263            | -178.761280 | -281.698266 | -75.824293  |
| undersampled_t_lr_k1 | response_model   | 20.000000  | 40000      | -0.000118       | 0.000102            | -23.626224  | -63.482972  | 16.230524   |
| random_targeting     | response_model   | 30.000000  | 60000      | -0.000790       | 0.000245            | -157.925372 | -253.875660 | -61.975084  |
| undersampled_t_lr_k1 | response_model   | 30.000000  | 60000      | -0.000195       | 0.000088            | -39.094992  | -73.651610  | -4.538373   |

At the pre-specified `5.00%` budget, the locked test
showed a negative advantage relative to response targeting. The estimated difference is `-195.6689`
incremental conversion outcomes with a confidence interval of
`[-295.0543, -96.2835]`.

## Ranking Metrics

| policy               | benchmark_relative_auuc |
| -------------------- | ----------------------- |
| response_model       | 0.001181                |
| undersampled_t_lr_k1 | 0.001079                |
| random_targeting     | 0.000629                |

AUUC is secondary. The decision is based on the budget-specific AIPW policy
contrast because the campaign has a fixed operating budget.

## Runtime

| stage            | model                   | fit_seconds |
| ---------------- | ----------------------- | ----------- |
| selection_fold_1 | random_targeting        | 0.000048    |
| selection_fold_1 | response_model          | 12.549045   |
| selection_fold_1 | undersampled_t_lr_k1    | 2.052309    |
| selection_fold_1 | aipw_nuisance_t_learner | 10.720724   |
| selection_fold_2 | random_targeting        | 0.000031    |
| selection_fold_2 | response_model          | 5.832194    |
| selection_fold_2 | undersampled_t_lr_k1    | 2.131176    |
| selection_fold_2 | aipw_nuisance_t_learner | 10.577988   |
| selection_fold_3 | random_targeting        | 0.000031    |
| selection_fold_3 | response_model          | 6.284495    |
| selection_fold_3 | undersampled_t_lr_k1    | 2.325398    |
| selection_fold_3 | aipw_nuisance_t_learner | 10.909834   |
| locked_test      | response_model          | 8.116473    |
| locked_test      | random_targeting        | 0.000038    |
| locked_test      | undersampled_t_lr_k1    | 3.235479    |
| locked_test      | aipw_nuisance_t_learner | 15.653483   |

## Statistical Scope

The AIPW intervals account for evaluation-sample uncertainty conditional on the
locked fitted policies. Repeated honest splits are required to quantify training
and selection instability. No production-impact claim is made without a live
randomized experiment.

## Reproducible Outputs

- Selection values: `outputs/tables/audit_conversion_selection.csv`
- Locked-test values: `outputs/tables/audit_conversion_test.csv`
- Paired contrasts: `outputs/tables/audit_conversion_contrasts.csv`
