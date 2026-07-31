# Uplift-Score Calibration on an Independent Holdout

## Setup

- Data: `data/processed/criteo_sample_2m.parquet` (2,000,000 rows), outcome `conversion`.
- Train/calibration/test rows: 1,200,000 / 400,000 / 400,000.
- Treatment propensity estimated from training data: `0.849933`.
- Model: `undersampled_t_lr_k1,undersampled_t_lr_k5`; random seed `42`.
- The isotonic calibrator is fitted only on the calibration set using transformed outcomes.
- The calibrator is fitted on `100` weighted quantile groups
  to reduce pseudo-outcome noise.
- Calibration metrics and threshold policies are evaluated only on the untouched test holdout.

## Calibration Results Summary

| model                | score_version | weighted_bias | weighted_mae | weighted_rmse | euce     | muce     | calibration_intercept | calibration_slope | benchmark_relative_auuc | score_mean | score_min | score_max |
| -------------------- | ------------- | ------------- | ------------ | ------------- | -------- | -------- | --------------------- | ----------------- | ----------------------- | ---------- | --------- | --------- |
| undersampled_t_lr_k1 | raw           | 0.000014      | 0.000224     | 0.000334      | 0.000224 | 0.000795 | -0.000104             | 1.140019          | 0.001021                | 0.000846   | -0.061297 | 0.184993  |
| undersampled_t_lr_k1 | calibrated    | -0.000172     | 0.000375     | 0.000661      | 0.000375 | 0.001909 | 0.000052              | 0.783273          | 0.001023                | 0.001032   | -0.006265 | 0.045925  |
| undersampled_t_lr_k5 | raw           | -0.000133     | 0.000369     | 0.000609      | 0.000369 | 0.001696 | 0.000084              | 0.783925          | 0.000961                | 0.001006   | -0.134885 | 0.488210  |
| undersampled_t_lr_k5 | calibrated    | -0.000106     | 0.000361     | 0.000603      | 0.000361 | 0.001690 | 0.000109              | 0.779587          | 0.000978                | 0.000979   | 0.000013  | 0.039269  |

Ideal calibration has an intercept near `0`, a slope near `1`, and errors near
`0`. Based on weighted MAE on the holdout, calibration improves: `undersampled_t_lr_k5`.
Isotonic mapping is monotonic and therefore preserves ordering in principle;
AUUC may change slightly because multiple scores can be mapped to the same value.

![Calibration plot](figures/conversion_uplift_calibration.png)

## Post-Calibration Groups

| model                | bin | n     | predicted_uplift | observed_uplift | ci_lower  | ci_upper |
| -------------------- | --- | ----- | ---------------- | --------------- | --------- | -------- |
| undersampled_t_lr_k1 | 1   | 40000 | 0.009180         | 0.007271        | 0.003435  | 0.011106 |
| undersampled_t_lr_k1 | 2   | 40000 | 0.000829         | 0.000674        | -0.000498 | 0.001847 |
| undersampled_t_lr_k1 | 3   | 40000 | 0.000567         | 0.000672        | 0.000104  | 0.001240 |
| undersampled_t_lr_k1 | 4   | 40000 | 0.000352         | -0.000311       | -0.000881 | 0.000259 |
| undersampled_t_lr_k1 | 5   | 40000 | -0.000041        | 0.000042        | -0.000314 | 0.000398 |
| undersampled_t_lr_k1 | 6   | 40000 | -0.000046        | 0.000148        | 0.000018  | 0.000277 |
| undersampled_t_lr_k1 | 7   | 40000 | -0.000046        | 0.000059        | -0.000023 | 0.000141 |
| undersampled_t_lr_k1 | 8   | 40000 | -0.000046        | 0.000059        | -0.000023 | 0.000141 |
| undersampled_t_lr_k1 | 9   | 40000 | -0.000046        | -0.000053       | -0.000405 | 0.000300 |
| undersampled_t_lr_k1 | 10  | 40000 | -0.000382        | 0.000039        | -0.001068 | 0.001145 |
| undersampled_t_lr_k5 | 1   | 40000 | 0.008118         | 0.006427        | 0.002452  | 0.010403 |
| undersampled_t_lr_k5 | 2   | 40000 | 0.000772         | 0.001230        | 0.000200  | 0.002260 |
| undersampled_t_lr_k5 | 3   | 40000 | 0.000695         | 0.000227        | -0.000490 | 0.000945 |
| undersampled_t_lr_k5 | 4   | 40000 | 0.000128         | 0.000044        | -0.000311 | 0.000398 |
| undersampled_t_lr_k5 | 5   | 40000 | 0.000013         | -0.000017       | -0.000365 | 0.000330 |
| undersampled_t_lr_k5 | 6   | 40000 | 0.000013         | 0.000118        | 0.000002  | 0.000234 |
| undersampled_t_lr_k5 | 7   | 40000 | 0.000013         | -0.000049       | -0.000395 | 0.000297 |
| undersampled_t_lr_k5 | 8   | 40000 | 0.000013         | 0.000059        | -0.000023 | 0.000140 |
| undersampled_t_lr_k5 | 9   | 40000 | 0.000013         | 0.000118        | 0.000002  | 0.000233 |
| undersampled_t_lr_k5 | 10  | 40000 | 0.000013         | 0.000571        | 0.000026  | 0.001115 |

Bin 1 contains the group with the highest raw scores. The confidence interval is
a normal approximation for the difference between treatment and control rates
within each bin. All bins are stored at `outputs/tables/conversion_uplift_calibration_bins.csv`.

## Threshold Policy Under Assumed Unit Economics

> **These currency figures are a worked example, not a finding.** The outcome
> value and the cost per contact below are command-line placeholders, not
> measured business inputs. Nothing here supports a revenue or ROI claim; the
> section exists only to show how a calibrated score would be turned into a
> break-even decision rule once real unit economics are available.

- Assumed value of one incremental conversion: `100.00`.
- Assumed cost per targeting action: `5.00`.
- Implied break-even uplift threshold: `0.050000`.

| model                | score_version | score_threshold | target_rate_pct | n_targeted | incremental_outcome | net_value    | profitable |
| -------------------- | ------------- | --------------- | --------------- | ---------- | ------------------- | ------------ | ---------- |
| undersampled_t_lr_k1 | raw           | 0.050000        | 0.217500        | 870        | 54.684653           | 1118.465345  | True       |
| undersampled_t_lr_k1 | calibrated    | 0.050000        | 0.000000        | 0          | 0.000000            | 0.000000     | False      |
| undersampled_t_lr_k5 | raw           | 0.050000        | 0.312000        | 1248       | 43.223095           | -1917.690511 | False      |
| undersampled_t_lr_k5 | calibrated    | 0.050000        | 0.000000        | 0          | 0.000000            | 0.000000     | False      |

Only `calibrated` rows can carry an absolute interpretation, because a raw score
is not on the probability scale. While calibration stability is unproven, the
fixed 5% budget remains the safer operating rule for the online experiment.

## Runtime

| model                | model_fit_seconds | calibrator_fit_seconds |
| -------------------- | ----------------- | ---------------------- |
| undersampled_t_lr_k1 | 3.673441          | 0.320991               |
| undersampled_t_lr_k5 | 1.050647          | 0.313386               |

## Recommendations

- Use the calibration plot to validate score magnitude, not as a replacement
  for AUUC as a ranking metric.
- Do not select a threshold on the test holdout after reviewing its results.
- Lock the model, calibrator, and threshold before the randomized online experiment.
