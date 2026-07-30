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
| s_learner           | 5.000000   | 40000      | 0.004592                 | 0.000274            | 0.004054      | 0.005130      | 3673.542378         | 3243.444288 | 4103.640469 | 0.010614                | 903.764325             | 469.549543          | 1337.979107         |
| t_learner           | 5.000000   | 40000      | 0.004276                 | 0.000272            | 0.003744      | 0.004808      | 3420.743680         | 2994.874407 | 3846.612952 | 0.009438                | 650.965627             | 197.708915          | 1104.222339         |
| transformed_outcome | 5.000000   | 40000      | 0.004181                 | 0.000266            | 0.003660      | 0.004703      | 3345.152467         | 2927.941657 | 3762.363278 | 0.010794                | 575.374415             | 167.115027          | 983.633803          |
| x_learner           | 5.000000   | 40000      | 0.003985                 | 0.000270            | 0.003455      | 0.004515      | 3188.022588         | 2763.977705 | 3612.067472 | 0.009803                | 418.244535             | -25.935734          | 862.424804          |
| r_learner           | 5.000000   | 40000      | 0.003855                 | 0.000272            | 0.003323      | 0.004388      | 3084.330678         | 2658.076440 | 3510.584915 | 0.009308                | 314.552625             | -119.733614         | 748.838864          |
| dr_learner          | 5.000000   | 40000      | 0.003839                 | 0.000269            | 0.003312      | 0.004367      | 3071.478981         | 2649.506506 | 3493.451456 | 0.009876                | 301.700928             | -136.222148         | 739.624004          |
| cvt                 | 5.000000   | 40000      | 0.002022                 | 0.000193            | 0.001643      | 0.002400      | 1617.367652         | 1314.596471 | 1920.138833 | 0.007884                | -1152.410401           | -1581.584350        | -723.236452         |
| random_targeting    | 5.000000   | 40000      | 0.000280                 | 0.000115            | 0.000055      | 0.000504      | 223.756233          | 44.221594   | 403.290872  | 0.005887                | -2546.021820           | -3058.401814        | -2033.641825        |
| response_model      | 5.000000   | 40000      | 0.003462                 | 0.000321            | 0.002832      | 0.004092      | 2769.778053         | 2265.772822 | 3273.783284 | 0.011002                | nan                    | nan                 | nan                 |

Selected champion: **s_learner**.

## Locked-Test Policy Value

| policy           | budget_pct | n_targeted | incremental_outcome_rate | standard_error_rate | ci_lower_rate | ci_upper_rate | incremental_outcome | ci_lower    | ci_upper    |
| ---------------- | ---------- | ---------- | ------------------------ | ------------------- | ------------- | ------------- | ------------------- | ----------- | ----------- |
| response_model   | 5.000000   | 10000      | 0.003759                 | 0.000642            | 0.002501      | 0.005017      | 751.837508          | 500.284152  | 1003.390864 |
| response_model   | 10.000000  | 20000      | 0.006727                 | 0.000831            | 0.005098      | 0.008356      | 1345.362365         | 1019.573980 | 1671.150750 |
| response_model   | 20.000000  | 40000      | 0.008117                 | 0.000954            | 0.006247      | 0.009988      | 1623.486837         | 1249.371725 | 1997.601948 |
| response_model   | 30.000000  | 60000      | 0.008453                 | 0.000990            | 0.006513      | 0.010392      | 1690.527659         | 1302.620432 | 2078.434886 |
| random_targeting | 5.000000   | 10000      | 0.000561                 | 0.000238            | 0.000095      | 0.001027      | 112.161829          | 18.964899   | 205.358759  |
| random_targeting | 10.000000  | 20000      | 0.000921                 | 0.000331            | 0.000272      | 0.001570      | 184.230233          | 54.445724   | 314.014743  |
| random_targeting | 20.000000  | 40000      | 0.002114                 | 0.000462            | 0.001210      | 0.003019      | 422.859676          | 241.926032  | 603.793320  |
| random_targeting | 30.000000  | 60000      | 0.002982                 | 0.000562            | 0.001881      | 0.004082      | 596.345696          | 376.214673  | 816.476719  |
| s_learner        | 5.000000   | 10000      | 0.005092                 | 0.000558            | 0.003999      | 0.006185      | 1018.361313         | 799.744744  | 1236.977882 |
| s_learner        | 10.000000  | 20000      | 0.006994                 | 0.000770            | 0.005484      | 0.008503      | 1398.724487         | 1096.792678 | 1700.656297 |
| s_learner        | 20.000000  | 40000      | 0.008169                 | 0.000937            | 0.006334      | 0.010005      | 1633.873781         | 1266.749923 | 2000.997640 |
| s_learner        | 30.000000  | 60000      | 0.008303                 | 0.000976            | 0.006390      | 0.010217      | 1660.656663         | 1277.929968 | 2043.383358 |

![Locked-test policy value](figures/audit_visit_policy_value.png)

## Paired Contrast Against Response Targeting

| policy           | reference_policy | budget_pct | n_targeted | difference_rate | standard_error_rate | difference   | ci_lower     | ci_upper    |
| ---------------- | ---------------- | ---------- | ---------- | --------------- | ------------------- | ------------ | ------------ | ----------- |
| random_targeting | response_model   | 5.000000   | 10000      | -0.003198       | 0.000652            | -639.675679  | -895.221983  | -384.129374 |
| s_learner        | response_model   | 5.000000   | 10000      | 0.001333        | 0.000568            | 266.523805   | 43.815000    | 489.232610  |
| random_targeting | response_model   | 10.000000  | 20000      | -0.005806       | 0.000814            | -1161.132132 | -1480.270573 | -841.993690 |
| s_learner        | response_model   | 10.000000  | 20000      | 0.000267        | 0.000459            | 53.362122    | -126.681715  | 233.405960  |
| random_targeting | response_model   | 20.000000  | 40000      | -0.006003       | 0.000870            | -1200.627161 | -1541.563938 | -859.690385 |
| s_learner        | response_model   | 20.000000  | 40000      | 0.000052        | 0.000257            | 10.386945    | -90.258786   | 111.032675  |
| random_targeting | response_model   | 30.000000  | 60000      | -0.005471       | 0.000841            | -1094.181963 | -1423.923511 | -764.440415 |
| s_learner        | response_model   | 30.000000  | 60000      | -0.000149       | 0.000202            | -29.870996   | -108.973242  | 49.231250   |

At the pre-specified `5.00%` budget, the locked test
showed a negative advantage relative to response targeting. The estimated difference is `-639.6757`
incremental visit outcomes with a confidence interval of
`[-895.2220, -384.1294]`.

## Ranking Metrics

| policy           | benchmark_relative_auuc |
| ---------------- | ----------------------- |
| response_model   | 0.011249                |
| s_learner        | 0.011002                |
| random_targeting | 0.006344                |

AUUC is secondary. The decision is based on the budget-specific AIPW policy
contrast because the campaign has a fixed operating budget.

## Runtime

| stage            | model                   | fit_seconds |
| ---------------- | ----------------------- | ----------- |
| selection_fold_1 | random_targeting        | 0.003591    |
| selection_fold_1 | response_model          | 7.298906    |
| selection_fold_1 | s_learner               | 4.001418    |
| selection_fold_1 | t_learner               | 13.049650   |
| selection_fold_1 | x_learner               | 15.475720   |
| selection_fold_1 | cvt                     | 3.444507    |
| selection_fold_1 | transformed_outcome     | 0.260527    |
| selection_fold_1 | r_learner               | 34.487367   |
| selection_fold_1 | dr_learner              | 30.774889   |
| selection_fold_1 | aipw_nuisance_t_learner | 5.626411    |
| selection_fold_2 | random_targeting        | 0.000024    |
| selection_fold_2 | response_model          | 3.558328    |
| selection_fold_2 | s_learner               | 3.995768    |
| selection_fold_2 | t_learner               | 5.339398    |
| selection_fold_2 | x_learner               | 11.123580   |
| selection_fold_2 | cvt                     | 4.090789    |
| selection_fold_2 | transformed_outcome     | 0.304671    |
| selection_fold_2 | r_learner               | 42.461448   |
| selection_fold_2 | dr_learner              | 32.254826   |
| selection_fold_2 | aipw_nuisance_t_learner | 10.198949   |
| selection_fold_3 | random_targeting        | 0.000022    |
| selection_fold_3 | response_model          | 3.622514    |
| selection_fold_3 | s_learner               | 4.083321    |
| selection_fold_3 | t_learner               | 4.960751    |
| selection_fold_3 | x_learner               | 9.829965    |
| selection_fold_3 | cvt                     | 3.233900    |
| selection_fold_3 | transformed_outcome     | 0.238353    |
| selection_fold_3 | r_learner               | 26.707329   |
| selection_fold_3 | dr_learner              | 32.311950   |
| selection_fold_3 | aipw_nuisance_t_learner | 4.904158    |
| locked_test      | response_model          | 4.416024    |
| locked_test      | random_targeting        | 0.000036    |
| locked_test      | s_learner               | 4.699406    |
| locked_test      | aipw_nuisance_t_learner | 6.205578    |

## Statistical Scope

The AIPW intervals account for evaluation-sample uncertainty conditional on the
locked fitted policies. Repeated honest splits are required to quantify training
and selection instability. No production-impact claim is made without a live
randomized experiment.

## Reproducible Outputs

- Selection values: `outputs/tables/audit_visit_selection.csv`
- Locked-test values: `outputs/tables/audit_visit_test.csv`
- Paired contrasts: `outputs/tables/audit_visit_contrasts.csv`
