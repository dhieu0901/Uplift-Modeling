# Hillstrom Pipeline Warm-up

## Purpose

Hillstrom provides a small randomized dataset for verifying the full loading, modeling, ranking, and evaluation workflow before running Criteo.

| Item | Value |
|---|---|
| Treatment | `Mens E-Mail` |
| Control | `No E-Mail` |
| Outcome | `visit` |
| Test share | 30% |
| Seed | `42` |

The `Womens E-Mail` arm is shown descriptively but excluded from the binary modeling task.

## Experimental Groups

| Segment | Rows | Visit rate | Conversion rate | Average spend |
|---|---:|---:|---:|---:|
| Mens E-Mail | 21,307 | 0.182757 | 0.012531 | 1.422617 |
| No E-Mail | 21,306 | 0.106167 | 0.005726 | 0.652789 |
| Womens E-Mail | 21,387 | 0.151400 | 0.008837 | 1.077202 |

## Best Observed Policy by Budget

| Budget | Policy | Users targeted | Incremental visits | Visits per 1,000 targeted |
|---:|---|---:|---:|---:|
| 5% | S-learner | 639 | 69.67 | 109.03 |
| 10% | Response model | 1,278 | 119.09 | 93.19 |
| 20% | X-learner | 2,557 | 236.64 | 92.54 |
| 30% | X-learner | 3,835 | 332.11 | 86.60 |

## Approximate AUUC

| Policy | AUUC |
|---|---:|
| X-learner | 506.60 |
| T-learner | 502.35 |
| Response model | 491.65 |
| S-learner | 488.21 |
| Random | 466.51 |

## Interpretation

The warm-up confirms that all policies run through the same workflow. Rankings vary by budget, so decisions use both budget-level gains and full-range AUUC.

These results validate the pipeline, not the final policy. The binary task does not reproduce the original M/W models.
