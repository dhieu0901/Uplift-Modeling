# Honest Uplift Model Selection: Criteo conversion

## Locked Protocol

- Data: `data/processed/criteo_sample_2m.parquet` (2,000,000 rows), outcome `conversion`.
- Base train/validation/test fractions: `0.60` / `0.20` / `0.20`.
- Selection folds: `3` over
  `1,600,000` selection observations.
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

| policy                   | budget_pct | n_targeted | incremental_outcome_rate | standard_error_rate | ci_lower_rate | ci_upper_rate | incremental_outcome | ci_lower    | ci_upper    | benchmark_relative_auuc | difference_vs_response | difference_ci_lower | difference_ci_upper |
| ------------------------ | ---------- | ---------- | ------------------------ | ------------------- | ------------- | ------------- | ------------------- | ----------- | ----------- | ----------------------- | ---------------------- | ------------------- | ------------------- |
| undersampled_t_lr_k1     | 5.000000   | 80000      | 0.000638                 | 0.000084            | 0.000473      | 0.000803      | 1020.428148         | 756.744162  | 1284.112133 | 0.000997                | -42.252835             | -185.681555         | 101.175884          |
| undersampled_t_lr_k5     | 5.000000   | 80000      | 0.000639                 | 0.000083            | 0.000476      | 0.000801      | 1022.070741         | 761.871255  | 1282.270227 | 0.000926                | -40.610241             | -191.522407         | 110.301925          |
| undersampled_t_lr_k10    | 5.000000   | 80000      | 0.000593                 | 0.000079            | 0.000438      | 0.000747      | 948.058122          | 701.042971  | 1195.073274 | 0.000898                | -114.622860            | -284.104603         | 54.858882           |
| undersampled_t_lr_k50    | 5.000000   | 80000      | 0.000552                 | 0.000081            | 0.000392      | 0.000711      | 882.589086          | 627.191328  | 1137.986843 | 0.000833                | -180.091897            | -340.603528         | -19.580266          |
| undersampled_t_lr_k25    | 5.000000   | 80000      | 0.000510                 | 0.000081            | 0.000351      | 0.000668      | 815.285624          | 561.180878  | 1069.390369 | 0.000850                | -247.395359            | -411.956941         | -82.833778          |
| undersampled_cvt_lr_k50  | 5.000000   | 80000      | 0.000500                 | 0.000062            | 0.000378      | 0.000622      | 799.680215          | 604.776213  | 994.584218  | 0.000944                | -263.000768            | -489.358904         | -36.642631          |
| undersampled_t_lr_k200   | 5.000000   | 80000      | 0.000469                 | 0.000074            | 0.000324      | 0.000614      | 750.580673          | 518.000222  | 983.161125  | 0.000752                | -312.100309            | -507.273687         | -116.926931         |
| undersampled_cvt_lr_k100 | 5.000000   | 80000      | 0.000487                 | 0.000053            | 0.000384      | 0.000591      | 779.528714          | 613.827124  | 945.230304  | 0.000906                | -283.152269            | -533.481885         | -32.822653          |
| undersampled_t_lr_k100   | 5.000000   | 80000      | 0.000393                 | 0.000060            | 0.000276      | 0.000510      | 628.914942          | 442.060690  | 815.769195  | 0.000658                | -433.766040            | -674.036150         | -193.495930         |
| undersampled_cvt_lr_k25  | 5.000000   | 80000      | 0.000333                 | 0.000071            | 0.000195      | 0.000472      | 533.045115          | 311.376570  | 754.713660  | 0.000798                | -529.635868            | -735.522364         | -323.749372         |
| undersampled_cvt_lr_k10  | 5.000000   | 80000      | 0.000297                 | 0.000076            | 0.000149      | 0.000446      | 475.782199          | 238.671612  | 712.892787  | 0.000473                | -586.898783            | -765.804432         | -407.993134         |
| undersampled_cvt_lr_k200 | 5.000000   | 80000      | 0.000291                 | 0.000041            | 0.000211      | 0.000371      | 465.681806          | 338.376050  | 592.987563  | 0.000742                | -596.999176            | -871.339801         | -322.658552         |
| undersampled_cvt_lr_k5   | 5.000000   | 80000      | 0.000147                 | 0.000063            | 0.000023      | 0.000271      | 235.822931          | 37.583889   | 434.061974  | 0.000292                | -826.858052            | -1044.255914        | -609.460189         |
| random_targeting         | 5.000000   | 80000      | 0.000092                 | 0.000023            | 0.000047      | 0.000137      | 147.484565          | 75.666499   | 219.302632  | 0.000475                | -915.196417            | -1198.084526        | -632.308309         |
| undersampled_cvt_lr_k1   | 5.000000   | 80000      | 0.000027                 | 0.000050            | -0.000072     | 0.000126      | 42.674293           | -115.688337 | 201.036924  | 0.000193                | -1020.006690           | -1264.992345        | -775.021035         |
| response_model           | 5.000000   | 80000      | 0.000664                 | 0.000092            | 0.000484      | 0.000845      | 1062.680983         | 774.093915  | 1351.268051 | 0.001025                | nan                    | nan                 | nan                 |

Selected champion: **undersampled_t_lr_k1**.

## Locked-Test Policy Value

| policy               | budget_pct | n_targeted | incremental_outcome_rate | standard_error_rate | ci_lower_rate | ci_upper_rate | incremental_outcome | ci_lower   | ci_upper   |
| -------------------- | ---------- | ---------- | ------------------------ | ------------------- | ------------- | ------------- | ------------------- | ---------- | ---------- |
| response_model       | 5.000000   | 20000      | 0.000465                 | 0.000183            | 0.000107      | 0.000823      | 185.932228          | 42.615224  | 329.249232 |
| response_model       | 10.000000  | 40000      | 0.000623                 | 0.000191            | 0.000249      | 0.000997      | 249.209015          | 99.784880  | 398.633151 |
| response_model       | 20.000000  | 80000      | 0.000631                 | 0.000200            | 0.000238      | 0.001023      | 252.299400          | 95.351677  | 409.247122 |
| response_model       | 30.000000  | 120000     | 0.000739                 | 0.000204            | 0.000339      | 0.001139      | 295.503072          | 135.463783 | 455.542360 |
| random_targeting     | 5.000000   | 20000      | 0.000087                 | 0.000032            | 0.000025      | 0.000150      | 34.911354           | 9.881846   | 59.940862  |
| random_targeting     | 10.000000  | 40000      | 0.000149                 | 0.000045            | 0.000061      | 0.000237      | 59.537945           | 24.430323  | 94.645567  |
| random_targeting     | 20.000000  | 80000      | 0.000231                 | 0.000079            | 0.000077      | 0.000385      | 92.259592           | 30.709904  | 153.809280 |
| random_targeting     | 30.000000  | 120000     | 0.000352                 | 0.000106            | 0.000144      | 0.000559      | 140.739876          | 57.693324  | 223.786428 |
| undersampled_t_lr_k1 | 5.000000   | 20000      | 0.000630                 | 0.000163            | 0.000311      | 0.000949      | 252.002144          | 124.315213 | 379.689075 |
| undersampled_t_lr_k1 | 10.000000  | 40000      | 0.000735                 | 0.000178            | 0.000387      | 0.001083      | 294.108613          | 154.945340 | 433.271887 |
| undersampled_t_lr_k1 | 20.000000  | 80000      | 0.000821                 | 0.000185            | 0.000459      | 0.001184      | 328.495916          | 183.577467 | 473.414365 |
| undersampled_t_lr_k1 | 30.000000  | 120000     | 0.000845                 | 0.000190            | 0.000473      | 0.001217      | 338.120359          | 189.350314 | 486.890404 |

![Locked-test policy value](figures/rare_conversion_development.png)

## Paired Contrast Against Response Targeting

| policy               | reference_policy | budget_pct | n_targeted | difference_rate | standard_error_rate | difference  | ci_lower    | ci_upper   |
| -------------------- | ---------------- | ---------- | ---------- | --------------- | ------------------- | ----------- | ----------- | ---------- |
| random_targeting     | response_model   | 5.000000   | 20000      | -0.000378       | 0.000183            | -151.020873 | -294.185800 | -7.855947  |
| undersampled_t_lr_k1 | response_model   | 5.000000   | 20000      | 0.000165        | 0.000090            | 66.069916   | -4.344372   | 136.484205 |
| random_targeting     | response_model   | 10.000000  | 40000      | -0.000474       | 0.000188            | -189.671070 | -337.003703 | -42.338438 |
| undersampled_t_lr_k1 | response_model   | 10.000000  | 40000      | 0.000112        | 0.000075            | 44.899598   | -14.245360  | 104.044556 |
| random_targeting     | response_model   | 20.000000  | 80000      | -0.000400       | 0.000187            | -160.039808 | -306.869794 | -13.209821 |
| undersampled_t_lr_k1 | response_model   | 20.000000  | 80000      | 0.000190        | 0.000079            | 76.196516   | 14.351748   | 138.041285 |
| random_targeting     | response_model   | 30.000000  | 120000     | -0.000387       | 0.000175            | -154.763196 | -292.212922 | -17.313470 |
| undersampled_t_lr_k1 | response_model   | 30.000000  | 120000     | 0.000107        | 0.000080            | 42.617287   | -19.930734  | 105.165308 |

At the pre-specified `5.00%` budget, the locked test
was inconclusive relative to response targeting. The estimated difference is `66.0699`
incremental conversion outcomes with a confidence interval of
`[-4.3444, 136.4842]`.

## Ranking Metrics

| policy               | benchmark_relative_auuc |
| -------------------- | ----------------------- |
| undersampled_t_lr_k1 | 0.001037                |
| response_model       | 0.000996                |
| random_targeting     | 0.000572                |

AUUC is secondary. The decision is based on the budget-specific AIPW policy
contrast because the campaign has a fixed operating budget.

## Runtime

| stage            | model                    | fit_seconds |
| ---------------- | ------------------------ | ----------- |
| selection_fold_1 | random_targeting         | 0.000055    |
| selection_fold_1 | response_model           | 13.387008   |
| selection_fold_1 | undersampled_t_lr_k1     | 4.492013    |
| selection_fold_1 | undersampled_cvt_lr_k1   | 2.235500    |
| selection_fold_1 | undersampled_t_lr_k5     | 0.788553    |
| selection_fold_1 | undersampled_cvt_lr_k5   | 0.561309    |
| selection_fold_1 | undersampled_t_lr_k10    | 0.485955    |
| selection_fold_1 | undersampled_cvt_lr_k10  | 0.329144    |
| selection_fold_1 | undersampled_t_lr_k25    | 0.273346    |
| selection_fold_1 | undersampled_cvt_lr_k25  | 0.219923    |
| selection_fold_1 | undersampled_t_lr_k50    | 0.194411    |
| selection_fold_1 | undersampled_cvt_lr_k50  | 0.173844    |
| selection_fold_1 | undersampled_t_lr_k100   | 0.167860    |
| selection_fold_1 | undersampled_cvt_lr_k100 | 0.143873    |
| selection_fold_1 | undersampled_t_lr_k200   | 0.148071    |
| selection_fold_1 | undersampled_cvt_lr_k200 | 0.126398    |
| selection_fold_1 | aipw_nuisance_t_learner  | 15.008474   |
| selection_fold_2 | random_targeting         | 0.000044    |
| selection_fold_2 | response_model           | 8.473642    |
| selection_fold_2 | undersampled_t_lr_k1     | 3.995958    |
| selection_fold_2 | undersampled_cvt_lr_k1   | 2.250675    |
| selection_fold_2 | undersampled_t_lr_k5     | 0.787833    |
| selection_fold_2 | undersampled_cvt_lr_k5   | 0.558882    |
| selection_fold_2 | undersampled_t_lr_k10    | 0.473745    |
| selection_fold_2 | undersampled_cvt_lr_k10  | 0.340191    |
| selection_fold_2 | undersampled_t_lr_k25    | 0.272277    |
| selection_fold_2 | undersampled_cvt_lr_k25  | 0.232512    |
| selection_fold_2 | undersampled_t_lr_k50    | 0.194160    |
| selection_fold_2 | undersampled_cvt_lr_k50  | 0.157520    |
| selection_fold_2 | undersampled_t_lr_k100   | 0.159181    |
| selection_fold_2 | undersampled_cvt_lr_k100 | 0.138218    |
| selection_fold_2 | undersampled_t_lr_k200   | 0.151303    |
| selection_fold_2 | undersampled_cvt_lr_k200 | 0.130262    |
| selection_fold_2 | aipw_nuisance_t_learner  | 12.936319   |
| selection_fold_3 | random_targeting         | 0.000031    |
| selection_fold_3 | response_model           | 9.370321    |
| selection_fold_3 | undersampled_t_lr_k1     | 4.281468    |
| selection_fold_3 | undersampled_cvt_lr_k1   | 2.613814    |
| selection_fold_3 | undersampled_t_lr_k5     | 0.752628    |
| selection_fold_3 | undersampled_cvt_lr_k5   | 0.613465    |
| selection_fold_3 | undersampled_t_lr_k10    | 0.553586    |
| selection_fold_3 | undersampled_cvt_lr_k10  | 0.350228    |
| selection_fold_3 | undersampled_t_lr_k25    | 0.337198    |
| selection_fold_3 | undersampled_cvt_lr_k25  | 0.291000    |
| selection_fold_3 | undersampled_t_lr_k50    | 0.234687    |
| selection_fold_3 | undersampled_cvt_lr_k50  | 0.194919    |
| selection_fold_3 | undersampled_t_lr_k100   | 0.176964    |
| selection_fold_3 | undersampled_cvt_lr_k100 | 0.142554    |
| selection_fold_3 | undersampled_t_lr_k200   | 0.164671    |
| selection_fold_3 | undersampled_cvt_lr_k200 | 0.165566    |
| selection_fold_3 | aipw_nuisance_t_learner  | 14.290466   |
| locked_test      | response_model           | 11.480135   |
| locked_test      | random_targeting         | 0.000034    |
| locked_test      | undersampled_t_lr_k1     | 5.806967    |
| locked_test      | aipw_nuisance_t_learner  | 17.041575   |

## Statistical Scope

The AIPW intervals account for evaluation-sample uncertainty conditional on the
locked fitted policies. Repeated honest splits are required to quantify training
and selection instability. No production-impact claim is made without a live
randomized experiment.

## Reproducible Outputs

- Selection values: `outputs/tables/rare_conversion_selection.csv`
- Locked-test values: `outputs/tables/rare_conversion_internal_holdout.csv`
- Paired contrasts: `outputs/tables/rare_conversion_internal_contrasts.csv`
