# Honest Uplift Model Selection: Criteo visit

## Locked Protocol

- Data: `data/processed/criteo_audit_1m.parquet` (1,000,000 rows), outcome `visit`.
- Base train/validation/test fractions: `0.60` / `0.20` / `0.20`.
- Selection folds: `3` over `800,000` selection observations.
- Primary targeting budget: `5.00%`.
- Confidence level: `95.0%`.
- Candidate model and hyperparameter selection uses 3-fold out-of-fold predictions on the combined development sample; locked-test outcomes are excluded.
- The champion is the candidate with the largest paired AIPW lower confidence bound against response targeting at the primary budget.
- After selection, the champion and response baseline are refit on train + validation.
- Locked-test outcomes are opened once for the final comparison.

## Out-of-Sample Development Selection

| policy              | budget_pct | n_targeted | incremental_outcome_rate | standard_error_rate | ci_lower_rate | ci_upper_rate | incremental_outcome | ci_lower    | ci_upper    | benchmark_relative_auuc | difference_vs_response | difference_ci_lower | difference_ci_upper |
| ------------------- | ---------- | ---------- | ------------------------ | ------------------- | ------------- | ------------- | ------------------- | ----------- | ----------- | ----------------------- | ---------------------- | ------------------- | ------------------- |
| s_learner           | 5.000000   | 40000      | 0.004751                 | 0.000272            | 0.004217      | 0.005284      | 3800.555747         | 3373.838317 | 4227.273177 | 0.010716                | 1001.096880            | 565.301546          | 1436.892213         |
| t_learner           | 5.000000   | 40000      | 0.004267                 | 0.000272            | 0.003733      | 0.004801      | 3413.335511         | 2986.193477 | 3840.477546 | 0.009506                | 613.876644             | 160.284038          | 1067.469250         |
| transformed_outcome | 5.000000   | 40000      | 0.004181                 | 0.000266            | 0.003660      | 0.004703      | 3345.152467         | 2927.941657 | 3762.363278 | 0.010794                | 545.693600             | 136.310332          | 955.076868          |
| dr_learner          | 5.000000   | 40000      | 0.004081                 | 0.000267            | 0.003558      | 0.004603      | 3264.525486         | 2846.513705 | 3682.537268 | 0.009736                | 465.066619             | 27.502089           | 902.631149          |
| x_learner           | 5.000000   | 40000      | 0.003932                 | 0.000271            | 0.003401      | 0.004463      | 3145.527653         | 2720.427265 | 3570.628041 | 0.009803                | 346.068786             | -90.261194          | 782.398766          |
| r_learner           | 5.000000   | 40000      | 0.003818                 | 0.000270            | 0.003289      | 0.004347      | 3054.366039         | 2630.902717 | 3477.829361 | 0.009507                | 254.907172             | -181.972622         | 691.786965          |
| cvt                 | 5.000000   | 40000      | 0.001834                 | 0.000188            | 0.001465      | 0.002203      | 1467.354479         | 1172.062745 | 1762.646212 | 0.007708                | -1332.104389           | -1764.226496        | -899.982281         |
| response_model      | 5.000000   | 40000      | 0.003499                 | 0.000321            | 0.002870      | 0.004128      | 2799.458867         | 2296.133473 | 3302.784262 | 0.010995                | nan                    | nan                 | nan                 |

Selected champion: **s_learner**.

## Locked-Test Policy Value

| policy         | budget_pct | n_targeted | incremental_outcome_rate | standard_error_rate | ci_lower_rate | ci_upper_rate | incremental_outcome | ci_lower    | ci_upper    |
| -------------- | ---------- | ---------- | ------------------------ | ------------------- | ------------- | ------------- | ------------------- | ----------- | ----------- |
| response_model | 5.000000   | 10000      | 0.003907                 | 0.000639            | 0.002653      | 0.005160      | 781.307633          | 530.658697  | 1031.956570 |
| response_model | 10.000000  | 20000      | 0.006445                 | 0.000833            | 0.004812      | 0.008078      | 1289.070350         | 962.445994  | 1615.694707 |
| response_model | 20.000000  | 40000      | 0.008087                 | 0.000954            | 0.006217      | 0.009958      | 1617.460465         | 1243.382659 | 1991.538272 |
| response_model | 30.000000  | 60000      | 0.008449                 | 0.000990            | 0.006508      | 0.010389      | 1689.738521         | 1301.609356 | 2077.867685 |
| s_learner      | 5.000000   | 10000      | 0.004749                 | 0.000565            | 0.003642      | 0.005856      | 949.763873          | 728.392773  | 1171.134972 |
| s_learner      | 10.000000  | 20000      | 0.006777                 | 0.000769            | 0.005270      | 0.008284      | 1355.364632         | 1054.007548 | 1656.721716 |
| s_learner      | 20.000000  | 40000      | 0.008236                 | 0.000930            | 0.006413      | 0.010060      | 1647.269557         | 1282.527880 | 2012.011233 |
| s_learner      | 30.000000  | 60000      | 0.008449                 | 0.000975            | 0.006539      | 0.010360      | 1689.837808         | 1307.761647 | 2071.913969 |

![Locked-test policy value](figures/audit_visit_policy_value.png)

## Paired Contrast Against Response Targeting

| policy    | reference_policy | budget_pct | n_targeted | difference_rate | standard_error_rate | difference | ci_lower    | ci_upper   |
| --------- | ---------------- | ---------- | ---------- | --------------- | ------------------- | ---------- | ----------- | ---------- |
| s_learner | response_model   | 5.000000   | 10000      | 0.000842        | 0.000565            | 168.456240 | -53.048474  | 389.960953 |
| s_learner | response_model   | 10.000000  | 20000      | 0.000331        | 0.000462            | 66.294282  | -114.787716 | 247.376280 |
| s_learner | response_model   | 20.000000  | 40000      | 0.000149        | 0.000272            | 29.809092  | -76.628474  | 136.246657 |
| s_learner | response_model   | 30.000000  | 60000      | 0.000000        | 0.000213            | 0.099287   | -83.244818  | 83.443393  |

At the pre-specified `5.00%` budget, the locked test
was inconclusive relative to response targeting. The estimated difference is `168.4562`
incremental visit outcomes with a confidence interval of
`[-53.0485, 389.9610]`.

## Ranking Metrics

| policy         | benchmark_relative_auuc |
| -------------- | ----------------------- |
| response_model | 0.011249                |
| s_learner      | 0.010943                |

AUUC is secondary. The decision is based on the budget-specific AIPW policy
contrast because the campaign has a fixed operating budget.

## Runtime

| stage            | model                   | fit_seconds |
| ---------------- | ----------------------- | ----------- |
| selection_fold_1 | response_model          | 19.015203   |
| selection_fold_1 | s_learner               | 18.213497   |
| selection_fold_1 | t_learner               | 17.821607   |
| selection_fold_1 | x_learner               | 40.352311   |
| selection_fold_1 | cvt                     | 14.630710   |
| selection_fold_1 | transformed_outcome     | 0.353736    |
| selection_fold_1 | r_learner               | 108.753578  |
| selection_fold_1 | dr_learner              | 109.598682  |
| selection_fold_1 | aipw_nuisance_t_learner | 17.674526   |
| selection_fold_2 | response_model          | 14.415943   |
| selection_fold_2 | s_learner               | 17.695532   |
| selection_fold_2 | t_learner               | 17.859594   |
| selection_fold_2 | x_learner               | 39.536360   |
| selection_fold_2 | cvt                     | 14.232301   |
| selection_fold_2 | transformed_outcome     | 0.335893    |
| selection_fold_2 | r_learner               | 107.609439  |
| selection_fold_2 | dr_learner              | 123.937894  |
| selection_fold_2 | aipw_nuisance_t_learner | 22.504932   |
| selection_fold_3 | response_model          | 18.922662   |
| selection_fold_3 | s_learner               | 23.807149   |
| selection_fold_3 | t_learner               | 22.517221   |
| selection_fold_3 | x_learner               | 50.457516   |
| selection_fold_3 | cvt                     | 17.687146   |
| selection_fold_3 | transformed_outcome     | 0.562682    |
| selection_fold_3 | r_learner               | 146.122656  |
| selection_fold_3 | dr_learner              | 138.866283  |
| selection_fold_3 | aipw_nuisance_t_learner | 22.849528   |
| locked_test      | response_model          | 27.104614   |
| locked_test      | s_learner               | 36.104741   |
| locked_test      | aipw_nuisance_t_learner | 34.858336   |

## Statistical Scope

The AIPW intervals account for evaluation-sample uncertainty conditional on the
locked fitted policies. Repeated honest splits are required to quantify training
and selection instability. No production-impact claim is made without a live
randomized experiment.

## Reproducible Outputs

- Selection values: `reports/tables/audit_visit_selection.csv`
- Locked-test values: `reports/tables/audit_visit_test.csv`
- Paired contrasts: `reports/tables/audit_visit_contrasts.csv`
