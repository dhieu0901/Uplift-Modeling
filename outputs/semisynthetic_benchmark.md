# Semi-Synthetic Uplift Benchmark with Known CATE

## Data-Generating Process

- Covariates: `data/processed/criteo_sample_500k.parquet` (200,000 rows).
- Target control outcome rate: `0.0500`.
- Realized mean control response surface: `0.050000`.
- Realized average CATE: `0.014927`.
- CATE standard deviation: `0.006987`.
- Treatment propensity: `0.5000`.
- The response surfaces contain nonlinear terms and feature interactions.
- Treatment is randomized and both potential-outcome probabilities are known.

## Honest Selection

The model is selected only from out-of-sample development paired AIPW lower confidence bounds
against response targeting at the pre-specified `5.00%`
budget. The selected model is
**t_learner**. All candidates are refit on development data and evaluated
on test only to compare them against known ground truth; this does not change
the development-selected champion.

| policy              | budget_pct | n_targeted | incremental_outcome_rate | standard_error_rate | ci_lower_rate | ci_upper_rate | incremental_outcome | ci_lower   | ci_upper   | benchmark_relative_auuc | difference_vs_response | difference_ci_lower | difference_ci_upper |
| ------------------- | ---------- | ---------- | ------------------------ | ------------------- | ------------- | ------------- | ------------------- | ---------- | ---------- | ----------------------- | ---------------------- | ------------------- | ------------------- |
| t_learner           | 5.000000   | 8000       | 0.001326                 | 0.000281            | 0.000774      | 0.001877      | 212.091392          | 123.839790 | 300.342994 | 0.008279                | 8.693657               | -72.450580          | 89.837894           |
| s_learner           | 5.000000   | 8000       | 0.001293                 | 0.000275            | 0.000754      | 0.001832      | 206.856806          | 120.621609 | 293.092003 | 0.008597                | 3.459070               | -89.438714          | 96.356854           |
| cvt                 | 5.000000   | 8000       | 0.001352                 | 0.000243            | 0.000876      | 0.001829      | 216.365322          | 140.112848 | 292.617795 | 0.007863                | 12.967586              | -96.214301          | 122.149473          |
| dr_learner          | 5.000000   | 8000       | 0.001185                 | 0.000267            | 0.000662      | 0.001708      | 189.582988          | 105.908512 | 273.257465 | 0.008686                | -13.814747             | -108.062268         | 80.432774           |
| r_learner           | 5.000000   | 8000       | 0.001089                 | 0.000265            | 0.000569      | 0.001609      | 174.238411          | 91.033284  | 257.443539 | 0.008855                | -29.159324             | -124.947946         | 66.629298           |
| transformed_outcome | 5.000000   | 8000       | 0.001135                 | 0.000255            | 0.000634      | 0.001635      | 181.525099          | 101.457615 | 261.592584 | 0.009696                | -21.872636             | -134.776251         | 91.030979           |
| x_learner           | 5.000000   | 8000       | 0.000894                 | 0.000270            | 0.000365      | 0.001423      | 143.053678          | 58.396360  | 227.710996 | 0.008601                | -60.344058             | -151.692732         | 31.004616           |
| random_targeting    | 5.000000   | 8000       | 0.000590                 | 0.000256            | 0.000088      | 0.001091      | 94.367523           | 14.103363  | 174.631682 | 0.007414                | -109.030212            | -227.231207         | 9.170782            |
| response_model      | 5.000000   | 8000       | 0.001271                 | 0.000291            | 0.000700      | 0.001842      | 203.397735          | 112.072549 | 294.722922 | 0.008396                | nan                    | nan                 | nan                 |

## CATE Recovery

| policy              | pehe     | cate_mae | cate_bias | pearson  | spearman |
| ------------------- | -------- | -------- | --------- | -------- | -------- |
| oracle              | 0.000000 | 0.000000 | 0.000000  | 1.000000 | 1.000000 |
| transformed_outcome | 0.004314 | 0.003273 | 0.000292  | 0.827801 | 0.849647 |
| s_learner           | 0.009863 | 0.005848 | -0.000748 | 0.490889 | 0.606008 |
| x_learner           | 0.014090 | 0.008236 | 0.000202  | 0.405798 | 0.516308 |
| dr_learner          | 0.020879 | 0.010510 | 0.000359  | 0.291934 | 0.465965 |
| r_learner           | 0.021032 | 0.010547 | 0.000221  | 0.284262 | 0.460127 |
| t_learner           | 0.022616 | 0.012043 | 0.000214  | 0.269787 | 0.391012 |
| cvt                 | 0.046908 | 0.019908 | 0.000444  | 0.153156 | 0.399924 |
| response_model      | 0.053397 | 0.048916 | 0.048170  | 0.206840 | 0.299592 |

PEHE, CATE MAE, and bias evaluate score magnitude. Pearson and Spearman
correlations evaluate linear and rank recovery. The response model is included
as an operational ranking baseline, not as a calibrated CATE estimator.

## Exact Policy Value at the Primary Budget

| policy              | budget_pct | n_targeted | true_incremental_outcome | oracle_incremental_outcome | policy_regret | oracle_value_fraction |
| ------------------- | ---------- | ---------- | ------------------------ | -------------------------- | ------------- | --------------------- |
| oracle              | 5.000000   | 2000       | 61.608541                | 61.608541                  | 0.000000      | 1.000000              |
| transformed_outcome | 5.000000   | 2000       | 54.075869                | 61.608541                  | 7.532672      | 0.877733              |
| s_learner           | 5.000000   | 2000       | 46.038917                | 61.608541                  | 15.569624     | 0.747281              |
| x_learner           | 5.000000   | 2000       | 44.854882                | 61.608541                  | 16.753658     | 0.728063              |
| t_learner           | 5.000000   | 2000       | 43.544052                | 61.608541                  | 18.064489     | 0.706786              |
| cvt                 | 5.000000   | 2000       | 42.643503                | 61.608541                  | 18.965038     | 0.692169              |
| dr_learner          | 5.000000   | 2000       | 41.866177                | 61.608541                  | 19.742364     | 0.679552              |
| r_learner           | 5.000000   | 2000       | 40.917212                | 61.608541                  | 20.691329     | 0.664148              |
| response_model      | 5.000000   | 2000       | 40.122763                | 61.608541                  | 21.485778     | 0.651253              |
| random_targeting    | 5.000000   | 2000       | 29.785534                | 61.608541                  | 31.823007     | 0.483464              |

`policy_regret` is the exact difference from targeting the users with the
largest true CATE. `oracle_value_fraction` measures how much of the attainable
oracle gain each ranking captures.

## Observed AIPW Estimate at the Same Budget

| policy              | budget_pct | n_targeted | incremental_outcome_rate | standard_error_rate | ci_lower_rate | ci_upper_rate | incremental_outcome | ci_lower  | ci_upper   |
| ------------------- | ---------- | ---------- | ------------------------ | ------------------- | ------------- | ------------- | ------------------- | --------- | ---------- |
| x_learner           | 5.000000   | 2000       | 0.002224                 | 0.000545            | 0.001157      | 0.003292      | 88.976132           | 46.284601 | 131.667663 |
| s_learner           | 5.000000   | 2000       | 0.001812                 | 0.000553            | 0.000728      | 0.002896      | 72.479094           | 29.108180 | 115.850009 |
| response_model      | 5.000000   | 2000       | 0.001811                 | 0.000554            | 0.000724      | 0.002897      | 72.428182           | 28.970881 | 115.885484 |
| cvt                 | 5.000000   | 2000       | 0.001731                 | 0.000486            | 0.000779      | 0.002683      | 69.227231           | 31.147585 | 107.306877 |
| t_learner           | 5.000000   | 2000       | 0.001661                 | 0.000537            | 0.000608      | 0.002714      | 66.428423           | 24.303961 | 108.552884 |
| r_learner           | 5.000000   | 2000       | 0.001372                 | 0.000534            | 0.000327      | 0.002418      | 54.899243           | 13.064253 | 96.734233  |
| random_targeting    | 5.000000   | 2000       | 0.001300                 | 0.000542            | 0.000239      | 0.002362      | 52.014827           | 9.552035  | 94.477618  |
| transformed_outcome | 5.000000   | 2000       | 0.001142                 | 0.000519            | 0.000126      | 0.002159      | 45.682479           | 5.023674  | 86.341283  |
| dr_learner          | 5.000000   | 2000       | 0.001123                 | 0.000507            | 0.000130      | 0.002116      | 44.915530           | 5.185392  | 84.645668  |

This comparison checks whether the observed-data estimator and its uncertainty
lead to decisions that agree with the known response surfaces.

![Exact policy value](figures/semisynthetic_policy_truth.png)

## Reproducible Outputs

- CATE metrics: `outputs/tables/semisynthetic_cate_metrics.csv`
- Development selection: `outputs/tables/semisynthetic_selection.csv`
- Exact policy values: `outputs/tables/semisynthetic_policy_truth.csv`
