# Confirmatory Criteo Audit Sample

## Construction

- Raw source: `data/criteo-uplift-v2.1.csv.gz`.
- Excluded development samples: `data/processed/criteo_sample_500k.parquet,data/processed/criteo_sample_2m.parquet,data/processed/criteo_audit_1m.parquet`.
- Requested rows: `4,000,000`.
- Reservoir seed: `20260730`.
- All complete benchmark-column hashes present in an excluded sample were
  removed before reservoir sampling.

## Audit Summary

| n       | treatment_rate | visit_rate | conversion_rate |
| ------- | -------------- | ---------- | --------------- |
| 4000000 | 0.843803       | 0.049598   | 0.003038        |

## Disjointness Check

| excluded_sample                           | overlap_rows |
| ----------------------------------------- | ------------ |
| data/processed/criteo_sample_500k.parquet | 0            |
| data/processed/criteo_sample_2m.parquet   | 0            |
| data/processed/criteo_audit_1m.parquet    | 0            |

The audit sample is stored at `data/processed/criteo_confirm_4m.parquet`.
Duplicate-valued rows sharing an excluded hash are conservatively removed.
