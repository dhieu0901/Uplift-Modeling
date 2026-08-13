# Uplift-Score Calibration on an Independent Holdout

## Setup

- Data: `data/processed/criteo_sample_2m.parquet` (2,000,000 rows), outcome `conversion`.
- Train/calibration/test rows: 1,200,000 / 400,000 / 400,000.
- Treatment propensity estimated from training data: `0.849624`.
- Model: `undersampled_t_lr_k1,undersampled_t_lr_k5`; random seed `42`.
- The isotonic calibrator is fitted only on the calibration set using transformed outcomes.
- The calibrator is fitted on `100` weighted quantile groups
  to reduce pseudo-outcome noise.
- Calibration metrics and threshold policies are evaluated only on the untouched test holdout.

## Calibration Results Summary

| model                | score_version | weighted_bias | weighted_mae | weighted_rmse | euce     | muce     | calibration_intercept | calibration_slope | benchmark_relative_auuc | score_mean | score_min | score_max |
| -------------------- | ------------- | ------------- | ------------ | ------------- | -------- | -------- | --------------------- | ----------------- | ----------------------- | ---------- | --------- | --------- |
| undersampled_t_lr_k1 | raw           | 0.000028      | 0.000315     | 0.000491      | 0.000315 | 0.001121 | 0.000014              | 1.015006          | 0.000988                | 0.000937   | -0.174339 | 0.245254  |
| undersampled_t_lr_k1 | calibrated    | -0.000087     | 0.000477     | 0.000728      | 0.000477 | 0.001775 | 0.000124              | 0.799655          | 0.000988                | 0.001052   | -0.006562 | 0.060262  |
| undersampled_t_lr_k5 | raw           | -0.000179     | 0.000393     | 0.000668      | 0.000393 | 0.001845 | 0.000046              | 0.794200          | 0.001012                | 0.001093   | -0.123733 | 0.360017  |
| undersampled_t_lr_k5 | calibrated    | -0.000083     | 0.000403     | 0.000596      | 0.000403 | 0.001350 | 0.000088              | 0.828072          | 0.001007                | 0.000997   | -0.005149 | 0.067368  |

Ideal calibration has an intercept near `0`, a slope near `1`, and errors near
`0`. Based on weighted MAE on the holdout, calibration improves: no models.
Isotonic mapping is monotonic and therefore preserves ordering in principle;
AUUC may change slightly because multiple scores can be mapped to the same value.

![Calibration plot](figures/conversion_uplift_calibration.png)

## Post-Calibration Groups

| model                | bin | n     | predicted_uplift | observed_uplift | ci_lower  | ci_upper |
| -------------------- | --- | ----- | ---------------- | --------------- | --------- | -------- |
| undersampled_t_lr_k1 | 1   | 40000 | 0.009802         | 0.008027        | 0.004434  | 0.011621 |
| undersampled_t_lr_k1 | 2   | 40000 | 0.000636         | -0.000316       | -0.001523 | 0.000892 |
| undersampled_t_lr_k1 | 3   | 40000 | 0.000221         | 0.000535        | -0.000122 | 0.001192 |
| undersampled_t_lr_k1 | 4   | 40000 | 0.000031         | -0.000003       | -0.000495 | 0.000490 |
| undersampled_t_lr_k1 | 5   | 40000 | 0.000018         | 0.000266        | 0.000092  | 0.000440 |
| undersampled_t_lr_k1 | 6   | 40000 | 0.000017         | 0.000160        | -0.000215 | 0.000535 |
| undersampled_t_lr_k1 | 7   | 40000 | -0.000015        | -0.000077       | -0.000416 | 0.000262 |
| undersampled_t_lr_k1 | 8   | 40000 | -0.000041        | 0.000029        | -0.000028 | 0.000087 |
| undersampled_t_lr_k1 | 9   | 40000 | -0.000047        | 0.000118        | 0.000002  | 0.000233 |
| undersampled_t_lr_k1 | 10  | 40000 | -0.000105        | 0.000906        | -0.000517 | 0.002330 |
| undersampled_t_lr_k5 | 1   | 40000 | 0.008861         | 0.007512        | 0.003593  | 0.011430 |
| undersampled_t_lr_k5 | 2   | 40000 | 0.000801         | -0.000250       | -0.001399 | 0.000900 |
| undersampled_t_lr_k5 | 3   | 40000 | 0.000152         | 0.000649        | 0.000378  | 0.000920 |
| undersampled_t_lr_k5 | 4   | 40000 | 0.000073         | 0.000260        | -0.000264 | 0.000785 |
| undersampled_t_lr_k5 | 5   | 40000 | 0.000073         | 0.000237        | 0.000073  | 0.000401 |
| undersampled_t_lr_k5 | 6   | 40000 | 0.000073         | 0.000044        | -0.000311 | 0.000398 |
| undersampled_t_lr_k5 | 7   | 40000 | 0.000037         | 0.000039        | -0.000322 | 0.000400 |
| undersampled_t_lr_k5 | 8   | 40000 | -0.000017        | 0.000029        | -0.000028 | 0.000087 |
| undersampled_t_lr_k5 | 9   | 40000 | -0.000017        | 0.000147        | 0.000018  | 0.000276 |
| undersampled_t_lr_k5 | 10  | 40000 | -0.000062        | 0.000473        | -0.000256 | 0.001203 |

Bin 1 contains the group with the highest raw scores. The confidence interval is
a normal approximation for the difference between treatment and control rates
within each bin. All bins are stored at `outputs/tables/conversion_uplift_calibration_bins.csv`.

## Break-Even Targeting at a Given Value Ratio

A calibrated score is on the outcome scale, so it can be compared against a
threshold instead of a rank. The threshold depends only on the ratio of what an
incremental conversion is worth to what a contact costs, which is why the
two enter as arguments: change the ratio and the section re-derives the rule.

- Value of one incremental conversion: `100.00`.
- Cost per targeting action: `5.00`.
- Break-even uplift threshold: `0.050000`.

| model                | score_version | score_threshold | target_rate_pct | n_targeted | incremental_outcome | net_value   | profitable |
| -------------------- | ------------- | --------------- | --------------- | ---------- | ------------------- | ----------- | ---------- |
| undersampled_t_lr_k1 | raw           | 0.050000        | 0.267750        | 1071       | 54.836720           | 128.671988  | True       |
| undersampled_t_lr_k1 | calibrated    | 0.050000        | 0.436750        | 1747       | 95.261267           | 791.126665  | True       |
| undersampled_t_lr_k5 | raw           | 0.050000        | 0.402750        | 1611       | 96.750000           | 1620.000000 | True       |
| undersampled_t_lr_k5 | calibrated    | 0.050000        | 0.468750        | 1875       | 101.037425          | 728.742454  | True       |

Only `calibrated` rows can carry an absolute interpretation, because a raw score
is not on the probability scale. The fixed 5% budget remains the operating rule
for the online experiment, because a threshold rule moves the number of users
contacted with the score distribution while a budget rule does not.

## Runtime

| model                | model_fit_seconds | calibrator_fit_seconds |
| -------------------- | ----------------- | ---------------------- |
| undersampled_t_lr_k1 | 3.158928          | 0.227625               |
| undersampled_t_lr_k5 | 0.721497          | 0.227107               |

## Recommendations

- Use the calibration plot to validate score magnitude, not as a replacement
  for AUUC as a ranking metric.
- Do not select a threshold on the test holdout after reviewing its results.
- Lock the model, calibrator, and threshold before the randomized online experiment.
