# Honest Uplift Model Selection: Criteo conversion

## Locked Protocol

- Data: `data/processed/criteo_audit_1m.parquet` (1,000,000 rows), outcome `conversion`.
- Base train/validation/test fractions: `0.60` / `0.20` / `0.20`.
- Selection folds: `3` over
  `800,000` selection observations.
- Primary targeting budget: `5.00%`.
- Confidence level: `95.0%`.
- Random seed: `777`. Every model seed is derived from it
  and the model's name, so the run is a function of this number alone.
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

| policy               | budget_pct | n_targeted | incremental_outcome_rate | standard_error_rate | ci_lower_rate | ci_upper_rate | incremental_outcome | ci_lower   | ci_upper   | benchmark_relative_auuc | difference_vs_response | difference_ci_lower | difference_ci_upper | ci_lower_adjusted |
| -------------------- | ---------- | ---------- | ------------------------ | ------------------- | ------------- | ------------- | ------------------- | ---------- | ---------- | ----------------------- | ---------------------- | ------------------- | ------------------- | ----------------- |
| undersampled_t_lr_k5 | 5.000000   | 40000      | 0.000800                 | 0.000115            | 0.000573      | 0.001026      | 639.831175          | 458.759631 | 820.902718 | 0.001087                | -59.333336             | -162.254495         | 43.587822           | -162.254495       |
| random_targeting     | 5.000000   | 40000      | 0.000074                 | 0.000029            | 0.000016      | 0.000132      | 59.297509           | 13.082656  | 105.512362 | 0.000620                | -639.867002            | -836.954913         | -442.779091         | nan               |
| response_model       | 5.000000   | 40000      | 0.000874                 | 0.000126            | 0.000626      | 0.001122      | 699.164511          | 500.825067 | 897.503954 | 0.001189                | nan                    | nan                 | nan                 | nan               |

Selected champion: **undersampled_t_lr_k5**.

`ci_lower_adjusted` re-derives each bound at
`95.00%`, which spreads the same
`5%` error rate across the
`1` candidates. The unadjusted column answers "is this candidate
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
| response_model       | 5.000000   | 10000      | 0.000963                 | 0.000271            | 0.000432      | 0.001493      | 192.567868          | 86.486894  | 298.648842 |
| response_model       | 10.000000  | 20000      | 0.001139                 | 0.000282            | 0.000586      | 0.001692      | 227.768470          | 117.223019 | 338.313921 |
| response_model       | 20.000000  | 40000      | 0.001198                 | 0.000292            | 0.000626      | 0.001769      | 239.507563          | 125.144088 | 353.871038 |
| response_model       | 30.000000  | 60000      | 0.001194                 | 0.000294            | 0.000618      | 0.001771      | 238.869870          | 123.563649 | 354.176090 |
| random_targeting     | 5.000000   | 10000      | -0.000013                | 0.000066            | -0.000141     | 0.000116      | -2.573075           | -28.250375 | 23.104224  |
| random_targeting     | 10.000000  | 20000      | -0.000014                | 0.000100            | -0.000211     | 0.000183      | -2.807603           | -42.148911 | 36.533705  |
| random_targeting     | 20.000000  | 40000      | 0.000070                 | 0.000128            | -0.000181     | 0.000321      | 13.954955           | -36.194289 | 64.104200  |
| random_targeting     | 30.000000  | 60000      | -0.000024                | 0.000169            | -0.000355     | 0.000306      | -4.887022           | -70.954506 | 61.180463  |
| undersampled_t_lr_k5 | 5.000000   | 10000      | 0.000964                 | 0.000240            | 0.000494      | 0.001434      | 192.776579          | 98.719695  | 286.833463 |
| undersampled_t_lr_k5 | 10.000000  | 20000      | 0.001128                 | 0.000258            | 0.000623      | 0.001634      | 225.699667          | 124.508194 | 326.891141 |
| undersampled_t_lr_k5 | 20.000000  | 40000      | 0.001224                 | 0.000273            | 0.000689      | 0.001758      | 244.709540          | 137.740649 | 351.678432 |
| undersampled_t_lr_k5 | 30.000000  | 60000      | 0.001195                 | 0.000280            | 0.000647      | 0.001743      | 239.006063          | 129.355158 | 348.656969 |

![Locked-test policy value](figures/audit_conversion_policy_value.png)

## Paired Contrast Against Response Targeting

| policy               | reference_policy | budget_pct | n_targeted | difference_rate | standard_error_rate | difference  | ci_lower    | ci_upper    |
| -------------------- | ---------------- | ---------- | ---------- | --------------- | ------------------- | ----------- | ----------- | ----------- |
| random_targeting     | response_model   | 5.000000   | 10000      | -0.000976       | 0.000268            | -195.140943 | -300.060632 | -90.221254  |
| undersampled_t_lr_k5 | response_model   | 5.000000   | 10000      | 0.000001        | 0.000132            | 0.208711    | -51.520892  | 51.938315   |
| random_targeting     | response_model   | 10.000000  | 20000      | -0.001153       | 0.000268            | -230.576072 | -335.718788 | -125.433357 |
| undersampled_t_lr_k5 | response_model   | 10.000000  | 20000      | -0.000010       | 0.000127            | -2.068802   | -51.727496  | 47.589891   |
| random_targeting     | response_model   | 20.000000  | 40000      | -0.001128       | 0.000262            | -225.552607 | -328.415173 | -122.690042 |
| undersampled_t_lr_k5 | response_model   | 20.000000  | 40000      | 0.000026        | 0.000105            | 5.201978    | -35.929175  | 46.333130   |
| random_targeting     | response_model   | 30.000000  | 60000      | -0.001219       | 0.000246            | -243.756891 | -340.248024 | -147.265759 |
| undersampled_t_lr_k5 | response_model   | 30.000000  | 60000      | 0.000001        | 0.000093            | 0.136194    | -36.150568  | 36.422956   |

At the pre-specified `5.00%` budget, the locked test
was inconclusive relative to response targeting. The estimated difference is `0.2087`
incremental conversion outcomes with a confidence interval of
`[-51.5209, 51.9383]`.

## Ranking Metrics

| policy               | benchmark_relative_auuc |
| -------------------- | ----------------------- |
| response_model       | 0.001205                |
| undersampled_t_lr_k5 | 0.001203                |
| random_targeting     | 0.000567                |

AUUC is secondary. The decision is based on the budget-specific AIPW policy
contrast because the campaign has a fixed operating budget.

## Runtime

| stage            | model                   | fit_seconds |
| ---------------- | ----------------------- | ----------- |
| selection_fold_1 | random_targeting        | 0.000049    |
| selection_fold_1 | response_model          | 6.521307    |
| selection_fold_1 | undersampled_t_lr_k5    | 0.334283    |
| selection_fold_1 | aipw_nuisance_t_learner | 4.342597    |
| selection_fold_2 | random_targeting        | 0.000056    |
| selection_fold_2 | response_model          | 2.553638    |
| selection_fold_2 | undersampled_t_lr_k5    | 0.329160    |
| selection_fold_2 | aipw_nuisance_t_learner | 4.213593    |
| selection_fold_3 | random_targeting        | 0.000023    |
| selection_fold_3 | response_model          | 3.413927    |
| selection_fold_3 | undersampled_t_lr_k5    | 0.323914    |
| selection_fold_3 | aipw_nuisance_t_learner | 4.575421    |
| locked_test      | response_model          | 4.493648    |
| locked_test      | random_targeting        | 0.000037    |
| locked_test      | undersampled_t_lr_k5    | 0.542048    |
| locked_test      | aipw_nuisance_t_learner | 6.503048    |

## Statistical Scope

The AIPW intervals account for evaluation-sample uncertainty conditional on the
locked fitted policies. Repeated honest splits are required to quantify training
and selection instability. No production-impact claim is made without a live
randomized experiment.

## Reproducible Outputs

- Selection values: `outputs/tables/audit_conversion_selection.csv`
- Locked-test values: `outputs/tables/audit_conversion_test.csv`
- Paired contrasts: `outputs/tables/audit_conversion_contrasts.csv`
