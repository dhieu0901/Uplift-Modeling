# End-to-End Honest-Split Stability: Criteo visit

## Protocol

- Data: `data/processed/criteo_sample_500k.parquet` (500,000 rows).
- Seeds: `42,123,2026,730,991,1201,1601,2401,3301,4401`.
- Primary budget: `5.00%`.
- Candidate models: `response_model,s_learner,t_learner,x_learner,cvt,transformed_outcome,r_learner,dr_learner`.
- Every run repeats training, out-of-sample model selection, development
  refitting, nuisance estimation, and locked-test evaluation.
- Each run uses the same pre-specified candidate set and selection rule.

## Aggregate Stability

| runs | mean_difference | std_difference | min_difference | max_difference | positive_point_rate | positive_ci_rate | negative_ci_rate |
| ---- | --------------- | -------------- | -------------- | -------------- | ------------------- | ---------------- | ---------------- |
| 10   | 69.235452       | 60.876559      | -10.816453     | 171.818011     | 0.800000            | 0.100000         | 0.000000         |

## Champion Frequency

| champion            | runs | mean_difference | positive_rate |
| ------------------- | ---- | --------------- | ------------- |
| s_learner           | 6    | 57.625832       | 0.666667      |
| transformed_outcome | 2    | 71.701318       | 1.000000      |
| x_learner           | 1    | 149.124736      | 1.000000      |
| dr_learner          | 1    | 54.072150       | 1.000000      |

## How Close Was Each Selection

- Median margin between the champion and the runner-up: `32.0` incremental outcomes.
- Runs where `s_learner` was not selected: **4 of 10**.
- Median half-width of the champion's own selection interval: `163.7`. The margin is `0.20` times that width.
- Runs in which no candidate reached a positive selection bound: **3 of 10**. The rule still names a champion in those runs, because it ranks candidates rather than requiring one to clear a bar.
- `s_learner` finished first or second in **7 of 10** runs (median rank 1).

The gap between first and second place is smaller than the uncertainty attached to first place itself, so the leaderboard reorders under resampling without any candidate being measurably better. A changing champion here reflects candidates the selection sample cannot separate, and the frequency table should be read as a ranking tendency rather than as evidence that one architecture wins.

## Results by Split

| seed | champion            | runner_up           | selection_margin | champion_selection_ci_lower | runner_up_selection_ci_lower | champion_selection_halfwidth | n_candidates_with_positive_bound | s_learner_selection_rank | difference_vs_response | ci_lower    | ci_upper   | champion_incremental_outcome | response_incremental_outcome | champion_auuc | response_auuc | fit_seconds |
| ---- | ------------------- | ------------------- | ---------------- | --------------------------- | ---------------------------- | ---------------------------- | -------------------------------- | ------------------------ | ---------------------- | ----------- | ---------- | ---------------------------- | ---------------------------- | ------------- | ------------- | ----------- |
| 42   | s_learner           | r_learner           | 25.586870        | 35.350220                   | 9.763350                     | 167.614987                   | 2                                | 1                        | 171.818011             | 9.287822    | 334.348201 | 499.147930                   | 327.329918                   | 0.009656      | 0.008810      | 223.066658  |
| 123  | s_learner           | x_learner           | 3.294823         | -69.338211                  | -72.633035                   | 169.632815                   | 0                                | 1                        | 111.294755             | -47.157707  | 269.747218 | 455.290688                   | 343.995932                   | 0.008762      | 0.008686      | 128.288874  |
| 2026 | s_learner           | transformed_outcome | 68.352085        | 22.325228                   | -46.026858                   | 178.228599                   | 1                                | 1                        | -8.241500              | -183.457789 | 166.974789 | 450.871741                   | 459.113241                   | 0.008648      | 0.008995      | 129.021052  |
| 730  | transformed_outcome | dr_learner          | 47.190695        | -16.951243                  | -64.141937                   | 160.239724                   | 0                                | 3                        | 85.284776              | -73.012979  | 243.582530 | 423.232921                   | 337.948146                   | 0.008975      | 0.008946      | 122.743778  |
| 991  | x_learner           | s_learner           | 14.477859        | -61.028969                  | -75.506828                   | 162.903318                   | 0                                | 2                        | 149.124736             | -7.870575   | 306.120048 | 346.572679                   | 197.447943                   | 0.008036      | 0.008481      | 77.165413   |
| 1201 | dr_learner          | x_learner           | 0.518596         | 39.210970                   | 38.692375                    | 155.836075                   | 5                                | 5                        | 54.072150              | -99.911692  | 208.055993 | 378.581729                   | 324.509579                   | 0.008158      | 0.008342      | 83.472002   |
| 1601 | s_learner           | r_learner           | 32.404664        | 38.949711                   | 6.545047                     | 151.087060                   | 2                                | 1                        | 36.075967              | -132.593529 | 204.745462 | 336.396354                   | 300.320387                   | 0.008828      | 0.008293      | 63.456410   |
| 2401 | s_learner           | dr_learner          | 31.564764        | 9.742574                    | -21.822190                   | 165.621482                   | 1                                | 1                        | -10.816453             | -179.798753 | 158.165846 | 434.843106                   | 445.659559                   | 0.007652      | 0.008396      | 65.008810   |
| 3301 | transformed_outcome | dr_learner          | 54.826029        | 83.846362                   | 29.020333                    | 159.384172                   | 3                                | 5                        | 58.117860              | -97.539924  | 213.775644 | 399.731745                   | 341.613885                   | 0.009269      | 0.008690      | 69.585575   |
| 4401 | s_learner           | t_learner           | 52.908189        | 152.492588                  | 99.584399                    | 164.483677                   | 4                                | 1                        | 45.624212              | -122.386857 | 213.635281 | 460.021639                   | 414.397427                   | 0.008889      | 0.008841      | 72.486649   |

![Honest-split stability](figures/visit_stability.png)

These repeated splits overlap and are therefore correlated robustness checks,
not independent experiments. They measure sensitivity to training and partition
variation; the canonical locked test remains the primary result.

Raw results: `outputs/tables/visit_stability.csv`.
