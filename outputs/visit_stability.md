# End-to-End Honest-Split Stability: Criteo visit

## Protocol

- Data: `data/processed/criteo_audit_1m.parquet` (1,000,000 rows).
- Seeds: `42,123,2026,730,991,1201,1601,2401,3301,4401`.
- Primary budget: `5.00%`.
- Candidate models: `response_model,s_learner,t_learner,x_learner,cvt,transformed_outcome,r_learner,dr_learner`.
- Selection: `3`-fold out-of-fold predictions over `800,000` observations.
- Every run repeats training, out-of-sample model selection, development
  refitting, nuisance estimation, and locked-test evaluation.
- Each run uses the same pre-specified candidate set and selection rule.

The selection stage is the one being measured, so it has to match the run that
produced the headline champion. A smaller selection sample widens every
candidate's interval, which would show up here as instability that belongs to
the sample size rather than to the rule.

## Aggregate Stability

| runs | mean_difference | std_difference | min_difference | max_difference | positive_point_rate | positive_ci_rate | negative_ci_rate |
| ---- | --------------- | -------------- | -------------- | -------------- | ------------------- | ---------------- | ---------------- |
| 10   | 163.426174      | 90.304500      | 44.232281      | 303.799070     | 1.000000            | 0.300000         | 0.000000         |

## Champion Frequency

| champion   | runs | mean_difference | positive_rate |
| ---------- | ---- | --------------- | ------------- |
| s_learner  | 8    | 183.176450      | 1.000000      |
| dr_learner | 2    | 84.425072       | 1.000000      |

## How Close Was Each Selection

- Median margin between the champion and the runner-up: `99.8` incremental outcomes.
- Runs where `s_learner` was not selected: **2 of 10**.
- Median half-width of the champion's own selection interval: `443.4`. The margin is `0.23` times that width.
- Every run had at least one candidate clear zero, a median of **6**. The rule ranks rather than requiring a bar to be cleared, so this is a property of the selection sample rather than something the rule enforces.
- `s_learner` finished first or second in **10 of 10** runs (median rank 1).

The gap between first and second place is smaller than the uncertainty attached to first place itself, so no candidate is measurably better than the one immediately below it. The ordering is not arbitrary either: `s_learner` never leaves the top two. Pairwise gaps inside the noise and a stable leader are consistent with each other, and together they say the sample can rank these candidates without being able to separate them. The defensible claim is about the policy class rather than about one architecture.

## Results by Split

| seed | champion   | runner_up           | selection_margin | champion_selection_ci_lower | runner_up_selection_ci_lower | champion_selection_halfwidth | n_candidates_with_positive_bound | s_learner_selection_rank | difference_vs_response | ci_lower    | ci_upper   | champion_incremental_outcome | response_incremental_outcome | champion_auuc | response_auuc | fit_seconds | sample_path                            | models                                                                                    | primary_budget | dataset_rows | selection_folds | selection_size |
| ---- | ---------- | ------------------- | ---------------- | --------------------------- | ---------------------------- | ---------------------------- | -------------------------------- | ------------------------ | ---------------------- | ----------- | ---------- | ---------------------------- | ---------------------------- | ------------- | ------------- | ----------- | -------------------------------------- | ----------------------------------------------------------------------------------------- | -------------- | ------------ | --------------- | -------------- |
| 42   | dr_learner | s_learner           | 98.397541        | 467.330268                  | 368.932727                   | 441.715877                   | 6                                | 2                        | 98.933323              | -126.153108 | 324.019753 | 997.813896                   | 898.880573                   | 0.009126      | 0.009990      | 688.069058  | data/processed/criteo_audit_1m.parquet | response_model,s_learner,t_learner,x_learner,cvt,transformed_outcome,r_learner,dr_learner | 0.050000       | 1000000      | 3               | 800000         |
| 123  | s_learner  | r_learner           | 354.105269       | 437.014044                  | 82.908775                    | 419.247834                   | 5                                | 1                        | 283.727273             | 75.586959   | 491.867586 | 840.199626                   | 556.472353                   | 0.009465      | 0.009549      | 592.728204  | data/processed/criteo_audit_1m.parquet | response_model,s_learner,t_learner,x_learner,cvt,transformed_outcome,r_learner,dr_learner | 0.050000       | 1000000      | 3               | 800000         |
| 2026 | s_learner  | x_learner           | 268.744443       | 497.588514                  | 228.844072                   | 446.365246                   | 4                                | 1                        | 185.347109             | -38.001827  | 408.696045 | 788.107235                   | 602.760125                   | 0.009954      | 0.009809      | 538.462950  | data/processed/criteo_audit_1m.parquet | response_model,s_learner,t_learner,x_learner,cvt,transformed_outcome,r_learner,dr_learner | 0.050000       | 1000000      | 3               | 800000         |
| 730  | s_learner  | dr_learner          | 80.364570        | 444.547615                  | 364.183046                   | 450.770590                   | 6                                | 1                        | 172.514826             | -55.321987  | 400.351639 | 885.666318                   | 713.151492                   | 0.009787      | 0.010028      | 579.035010  | data/processed/criteo_audit_1m.parquet | response_model,s_learner,t_learner,x_learner,cvt,transformed_outcome,r_learner,dr_learner | 0.050000       | 1000000      | 3               | 800000         |
| 991  | s_learner  | x_learner           | 60.288044        | 479.735373                  | 419.447329                   | 430.613248                   | 6                                | 1                        | 168.694074             | -48.764286  | 386.152434 | 794.137218                   | 625.443144                   | 0.009757      | 0.009964      | 469.689966  | data/processed/criteo_audit_1m.parquet | response_model,s_learner,t_learner,x_learner,cvt,transformed_outcome,r_learner,dr_learner | 0.050000       | 1000000      | 3               | 800000         |
| 1201 | s_learner  | dr_learner          | 133.177700       | 490.044329                  | 356.866630                   | 449.102927                   | 6                                | 1                        | 78.313221              | -146.852886 | 303.479328 | 904.562190                   | 826.248968                   | 0.009763      | 0.009982      | 359.001110  | data/processed/criteo_audit_1m.parquet | response_model,s_learner,t_learner,x_learner,cvt,transformed_outcome,r_learner,dr_learner | 0.050000       | 1000000      | 3               | 800000         |
| 1601 | s_learner  | transformed_outcome | 202.447736       | 565.178162                  | 362.730425                   | 445.155186                   | 5                                | 1                        | 44.232281              | -183.647401 | 272.111963 | 772.469094                   | 728.236813                   | 0.009714      | 0.009792      | 429.861926  | data/processed/criteo_audit_1m.parquet | response_model,s_learner,t_learner,x_learner,cvt,transformed_outcome,r_learner,dr_learner | 0.050000       | 1000000      | 3               | 800000         |
| 2401 | s_learner  | transformed_outcome | 101.282574       | 263.129656                  | 161.847082                   | 433.026443                   | 6                                | 1                        | 303.799070             | 95.486028   | 512.112112 | 871.134462                   | 567.335392                   | 0.009532      | 0.009735      | 239.204384  | data/processed/criteo_audit_1m.parquet | response_model,s_learner,t_learner,x_learner,cvt,transformed_outcome,r_learner,dr_learner | 0.050000       | 1000000      | 3               | 800000         |
| 3301 | dr_learner | s_learner           | 52.286765        | 515.875629                  | 463.588864                   | 430.597358                   | 6                                | 2                        | 69.916821              | -151.537777 | 291.371419 | 847.237460                   | 777.320640                   | 0.009073      | 0.009795      | 267.545682  | data/processed/criteo_audit_1m.parquet | response_model,s_learner,t_learner,x_learner,cvt,transformed_outcome,r_learner,dr_learner | 0.050000       | 1000000      | 3               | 800000         |
| 4401 | s_learner  | dr_learner          | 9.510515         | 376.859259                  | 367.348744                   | 446.843664                   | 5                                | 1                        | 228.783745             | 13.601290   | 443.966199 | 981.787865                   | 753.004120                   | 0.010008      | 0.010015      | 236.688966  | data/processed/criteo_audit_1m.parquet | response_model,s_learner,t_learner,x_learner,cvt,transformed_outcome,r_learner,dr_learner | 0.050000       | 1000000      | 3               | 800000         |

![Honest-split stability](figures/visit_stability.png)

These repeated splits overlap and are therefore correlated robustness checks,
not independent experiments. They measure sensitivity to training and partition
variation; the canonical locked test remains the primary result.

Raw results: `outputs/tables/visit_stability.csv`.
