# Confirmatory Criteo Audit Sample

## Construction

- Indexed source: `data/processed/criteo_indexed.parquet`.
- Excluded development samples: `data/processed/criteo_sample_500k.parquet`, `data/processed/criteo_audit_1m.parquet`, `data/processed/criteo_confirm_4m.parquet`.
- Requested rows: `2,000,000`.
- Reservoir seed: `42`.
- Rows already used by an excluded sample were removed by `row_id` before
  reservoir sampling, so exactly those rows are withheld and untouched
  duplicates of them remain eligible.

## Audit Summary

| n       | treatment_rate | visit_rate | conversion_rate |
| ------- | -------------- | ---------- | --------------- |
| 2000000 | 0.849625       | 0.047087   | 0.002907        |

## Disjointness Check

| excluded_sample                           | overlap_rows |
| ----------------------------------------- | ------------ |
| data/processed/criteo_sample_500k.parquet | 0            |
| data/processed/criteo_audit_1m.parquet    | 0            |
| data/processed/criteo_confirm_4m.parquet  | 0            |

The audit sample is stored at `data/processed/criteo_sample_2m_new.parquet`.
