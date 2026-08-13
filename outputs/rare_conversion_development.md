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

| policy                   | budget_pct | n_targeted | incremental_outcome_rate | standard_error_rate | ci_lower_rate | ci_upper_rate | incremental_outcome | ci_lower    | ci_upper    | benchmark_relative_auuc | difference_vs_response | difference_ci_lower | difference_ci_upper | ci_lower_adjusted |
| ------------------------ | ---------- | ---------- | ------------------------ | ------------------- | ------------- | ------------- | ------------------- | ----------- | ----------- | ----------------------- | ---------------------- | ------------------- | ------------------- | ----------------- |
| undersampled_t_lr_k5     | 5.000000   | 80000      | 0.000812                 | 0.000084            | 0.000648      | 0.000976      | 1299.238110         | 1036.506618 | 1561.969601 | 0.001071                | -0.153993              | -125.984084         | 125.676099          | -187.215823       |
| undersampled_t_lr_k10    | 5.000000   | 80000      | 0.000796                 | 0.000087            | 0.000626      | 0.000966      | 1273.901704         | 1001.823641 | 1545.979766 | 0.001063                | -25.490398             | -132.646903         | 81.666107           | -184.791657       |
| undersampled_t_lr_k1     | 5.000000   | 80000      | 0.000780                 | 0.000082            | 0.000619      | 0.000941      | 1247.812852         | 990.299124  | 1505.326580 | 0.001057                | -51.579250             | -189.110215         | 85.951715           | -256.035861       |
| undersampled_t_lr_k25    | 5.000000   | 80000      | 0.000764                 | 0.000077            | 0.000614      | 0.000914      | 1222.848065         | 982.830092  | 1462.866039 | 0.001020                | -76.544037             | -242.645266         | 89.557193           | -323.473842       |
| undersampled_t_lr_k50    | 5.000000   | 80000      | 0.000672                 | 0.000081            | 0.000515      | 0.000830      | 1075.906048         | 823.279394  | 1328.532702 | 0.000960                | -223.486054            | -370.519258         | -76.452851          | -442.068905       |
| undersampled_t_lr_k100   | 5.000000   | 80000      | 0.000671                 | 0.000068            | 0.000537      | 0.000805      | 1074.042451         | 859.519832  | 1288.565070 | 0.000898                | -225.349651            | -427.019040         | -23.680263          | -525.155878       |
| undersampled_cvt_lr_k50  | 5.000000   | 80000      | 0.000630                 | 0.000067            | 0.000500      | 0.000761      | 1008.578697         | 799.622273  | 1217.535122 | 0.000983                | -290.813405            | -493.581361         | -88.045448          | -592.252787       |
| undersampled_cvt_lr_k25  | 5.000000   | 80000      | 0.000606                 | 0.000076            | 0.000457      | 0.000755      | 969.484658          | 730.941921  | 1208.027396 | 0.000939                | -329.907444            | -498.458001         | -161.356887         | -580.478473       |
| undersampled_cvt_lr_k100 | 5.000000   | 80000      | 0.000622                 | 0.000057            | 0.000509      | 0.000734      | 995.012075          | 815.103576  | 1174.920574 | 0.001024                | -304.380027            | -530.474405         | -78.285650          | -640.496988       |
| undersampled_t_lr_k200   | 5.000000   | 80000      | 0.000590                 | 0.000074            | 0.000445      | 0.000736      | 944.476036          | 711.419697  | 1177.532375 | 0.000738                | -354.916066            | -537.493767         | -172.338366         | -626.340161       |
| undersampled_cvt_lr_k200 | 5.000000   | 80000      | 0.000459                 | 0.000044            | 0.000373      | 0.000545      | 734.688870          | 597.244322  | 872.133418  | 0.000954                | -564.703232            | -817.430286         | -311.976179         | -940.412925       |
| undersampled_cvt_lr_k10  | 5.000000   | 80000      | 0.000317                 | 0.000069            | 0.000182      | 0.000451      | 506.455182          | 291.006774  | 721.903590  | 0.000540                | -792.936921            | -982.248541         | -603.625300         | -1074.371811      |
| undersampled_cvt_lr_k5   | 5.000000   | 80000      | 0.000223                 | 0.000068            | 0.000089      | 0.000357      | 357.238944          | 142.518023  | 571.959866  | 0.000454                | -942.153158            | -1127.894560        | -756.411755         | -1218.280483      |
| undersampled_cvt_lr_k1   | 5.000000   | 80000      | 0.000064                 | 0.000054            | -0.000042     | 0.000170      | 102.242790          | -67.929281  | 272.414861  | 0.000253                | -1197.149312           | -1422.602372        | -971.696252         | -1532.312876      |
| random_targeting         | 5.000000   | 80000      | 0.000042                 | 0.000024            | -0.000005     | 0.000089      | 67.212838           | -7.521827   | 141.947503  | 0.000559                | -1232.179264           | -1503.123693        | -961.234834         | nan               |
| response_model           | 5.000000   | 80000      | 0.000812                 | 0.000089            | 0.000638      | 0.000986      | 1299.392102         | 1021.227411 | 1577.556793 | 0.001080                | nan                    | nan                 | nan                 | nan               |

Selected champion: **undersampled_t_lr_k5**.

`ci_lower_adjusted` re-derives each bound at
`99.64%`, which spreads the same
`5%` error rate across the
`14` candidates. The unadjusted column answers "is this candidate
above response targeting"; the adjusted one answers "does any candidate in this
table stand above it", which is the question the selection rule actually asks by
keeping the largest bound. No candidate clears zero once the table is read as a whole, so the champion is the best of a set that the selection sample cannot separate from response targeting.

Selection still uses the unadjusted rule fixed before the data was seen.
Adjusting penalizes wide intervals more than narrow ones and so can reorder the
table, and switching to it here would be choosing a rule after seeing the
result.

## Locked-Test Policy Value

| policy               | budget_pct | n_targeted | incremental_outcome_rate | standard_error_rate | ci_lower_rate | ci_upper_rate | incremental_outcome | ci_lower   | ci_upper   |
| -------------------- | ---------- | ---------- | ------------------------ | ------------------- | ------------- | ------------- | ------------------- | ---------- | ---------- |
| response_model       | 5.000000   | 20000      | 0.000711                 | 0.000180            | 0.000358      | 0.001064      | 284.310291          | 143.212889 | 425.407694 |
| response_model       | 10.000000  | 40000      | 0.000789                 | 0.000190            | 0.000416      | 0.001162      | 315.721824          | 166.444222 | 464.999426 |
| response_model       | 20.000000  | 80000      | 0.000811                 | 0.000198            | 0.000424      | 0.001199      | 324.551321          | 169.702588 | 479.400054 |
| response_model       | 30.000000  | 120000     | 0.000864                 | 0.000200            | 0.000472      | 0.001255      | 345.546828          | 188.973194 | 502.120462 |
| random_targeting     | 5.000000   | 20000      | 0.000057                 | 0.000043            | -0.000028     | 0.000142      | 22.661933           | -11.381388 | 56.705254  |
| random_targeting     | 10.000000  | 40000      | 0.000111                 | 0.000063            | -0.000012     | 0.000234      | 44.505083           | -4.754482  | 93.764648  |
| random_targeting     | 20.000000  | 80000      | 0.000064                 | 0.000097            | -0.000126     | 0.000253      | 25.416471           | -50.526097 | 101.359040 |
| random_targeting     | 30.000000  | 120000     | 0.000318                 | 0.000115            | 0.000093      | 0.000543      | 127.159062          | 37.114768  | 217.203356 |
| undersampled_t_lr_k5 | 5.000000   | 20000      | 0.000488                 | 0.000170            | 0.000155      | 0.000821      | 195.237933          | 62.155144  | 328.320722 |
| undersampled_t_lr_k5 | 10.000000  | 40000      | 0.000628                 | 0.000178            | 0.000279      | 0.000977      | 251.128852          | 111.625767 | 390.631938 |
| undersampled_t_lr_k5 | 20.000000  | 80000      | 0.000674                 | 0.000184            | 0.000314      | 0.001034      | 269.684558          | 125.584008 | 413.785107 |
| undersampled_t_lr_k5 | 30.000000  | 120000     | 0.000694                 | 0.000188            | 0.000325      | 0.001064      | 277.668850          | 129.906912 | 425.430787 |

![Locked-test policy value](figures/rare_conversion_development.png)

## Paired Contrast Against Response Targeting

| policy               | reference_policy | budget_pct | n_targeted | difference_rate | standard_error_rate | difference  | ci_lower    | ci_upper    |
| -------------------- | ---------------- | ---------- | ---------- | --------------- | ------------------- | ----------- | ----------- | ----------- |
| random_targeting     | response_model   | 5.000000   | 20000      | -0.000654       | 0.000176            | -261.648358 | -399.535384 | -123.761333 |
| undersampled_t_lr_k5 | response_model   | 5.000000   | 20000      | -0.000223       | 0.000075            | -89.072359  | -148.202690 | -29.942027  |
| random_targeting     | response_model   | 10.000000  | 40000      | -0.000678       | 0.000182            | -271.216741 | -413.705751 | -128.727731 |
| undersampled_t_lr_k5 | response_model   | 10.000000  | 40000      | -0.000161       | 0.000081            | -64.592972  | -128.297113 | -0.888831   |
| random_targeting     | response_model   | 20.000000  | 80000      | -0.000748       | 0.000176            | -299.134849 | -436.867661 | -161.402038 |
| undersampled_t_lr_k5 | response_model   | 20.000000  | 80000      | -0.000137       | 0.000074            | -54.866763  | -112.823638 | 3.090111    |
| random_targeting     | response_model   | 30.000000  | 120000     | -0.000546       | 0.000171            | -218.387766 | -352.075316 | -84.700217  |
| undersampled_t_lr_k5 | response_model   | 30.000000  | 120000     | -0.000170       | 0.000072            | -67.877978  | -124.400059 | -11.355898  |

At the pre-specified `5.00%` budget, the locked test
showed a negative advantage relative to response targeting. The estimated difference is `-89.0724`
incremental conversion outcomes with a confidence interval of
`[-148.2027, -29.9420]`.

## Ranking Metrics

| policy               | benchmark_relative_auuc |
| -------------------- | ----------------------- |
| response_model       | 0.001056                |
| undersampled_t_lr_k5 | 0.000956                |
| random_targeting     | 0.000483                |

AUUC is secondary. The decision is based on the budget-specific AIPW policy
contrast because the campaign has a fixed operating budget.

## Runtime

| stage            | model                    | fit_seconds |
| ---------------- | ------------------------ | ----------- |
| selection_fold_1 | random_targeting         | 0.000045    |
| selection_fold_1 | response_model           | 9.462337    |
| selection_fold_1 | undersampled_t_lr_k1     | 3.031514    |
| selection_fold_1 | undersampled_cvt_lr_k1   | 1.645083    |
| selection_fold_1 | undersampled_t_lr_k5     | 0.682950    |
| selection_fold_1 | undersampled_cvt_lr_k5   | 0.429468    |
| selection_fold_1 | undersampled_t_lr_k10    | 0.359747    |
| selection_fold_1 | undersampled_cvt_lr_k10  | 0.257890    |
| selection_fold_1 | undersampled_t_lr_k25    | 0.218502    |
| selection_fold_1 | undersampled_cvt_lr_k25  | 0.178443    |
| selection_fold_1 | undersampled_t_lr_k50    | 0.160782    |
| selection_fold_1 | undersampled_cvt_lr_k50  | 0.126930    |
| selection_fold_1 | undersampled_t_lr_k100   | 0.136722    |
| selection_fold_1 | undersampled_cvt_lr_k100 | 0.116657    |
| selection_fold_1 | undersampled_t_lr_k200   | 0.126270    |
| selection_fold_1 | undersampled_cvt_lr_k200 | 0.106353    |
| selection_fold_1 | aipw_nuisance_t_learner  | 7.701819    |
| selection_fold_2 | random_targeting         | 0.000033    |
| selection_fold_2 | response_model           | 5.724957    |
| selection_fold_2 | undersampled_t_lr_k1     | 2.932572    |
| selection_fold_2 | undersampled_cvt_lr_k1   | 1.718170    |
| selection_fold_2 | undersampled_t_lr_k5     | 0.664570    |
| selection_fold_2 | undersampled_cvt_lr_k5   | 0.428221    |
| selection_fold_2 | undersampled_t_lr_k10    | 0.334788    |
| selection_fold_2 | undersampled_cvt_lr_k10  | 0.247172    |
| selection_fold_2 | undersampled_t_lr_k25    | 0.212084    |
| selection_fold_2 | undersampled_cvt_lr_k25  | 0.187077    |
| selection_fold_2 | undersampled_t_lr_k50    | 0.162207    |
| selection_fold_2 | undersampled_cvt_lr_k50  | 0.130396    |
| selection_fold_2 | undersampled_t_lr_k100   | 0.132420    |
| selection_fold_2 | undersampled_cvt_lr_k100 | 0.114143    |
| selection_fold_2 | undersampled_t_lr_k200   | 0.127967    |
| selection_fold_2 | undersampled_cvt_lr_k200 | 0.106232    |
| selection_fold_2 | aipw_nuisance_t_learner  | 7.536186    |
| selection_fold_3 | random_targeting         | 0.000026    |
| selection_fold_3 | response_model           | 5.604640    |
| selection_fold_3 | undersampled_t_lr_k1     | 3.165470    |
| selection_fold_3 | undersampled_cvt_lr_k1   | 1.740049    |
| selection_fold_3 | undersampled_t_lr_k5     | 0.623348    |
| selection_fold_3 | undersampled_cvt_lr_k5   | 0.430841    |
| selection_fold_3 | undersampled_t_lr_k10    | 0.357977    |
| selection_fold_3 | undersampled_cvt_lr_k10  | 0.257846    |
| selection_fold_3 | undersampled_t_lr_k25    | 0.211301    |
| selection_fold_3 | undersampled_cvt_lr_k25  | 0.183504    |
| selection_fold_3 | undersampled_t_lr_k50    | 0.159442    |
| selection_fold_3 | undersampled_cvt_lr_k50  | 0.134660    |
| selection_fold_3 | undersampled_t_lr_k100   | 0.134368    |
| selection_fold_3 | undersampled_cvt_lr_k100 | 0.119856    |
| selection_fold_3 | undersampled_t_lr_k200   | 0.131025    |
| selection_fold_3 | undersampled_cvt_lr_k200 | 0.107618    |
| selection_fold_3 | aipw_nuisance_t_learner  | 7.243434    |
| locked_test      | response_model           | 8.024185    |
| locked_test      | random_targeting         | 0.000035    |
| locked_test      | undersampled_t_lr_k5     | 0.936085    |
| locked_test      | aipw_nuisance_t_learner  | 10.327556   |

## Statistical Scope

The AIPW intervals account for evaluation-sample uncertainty conditional on the
locked fitted policies. Repeated honest splits are required to quantify training
and selection instability. No production-impact claim is made without a live
randomized experiment.

## Reproducible Outputs

- Selection values: `outputs/tables/rare_conversion_selection.csv`
- Locked-test values: `outputs/tables/rare_conversion_internal_holdout.csv`
- Paired contrasts: `outputs/tables/rare_conversion_internal_contrasts.csv`
