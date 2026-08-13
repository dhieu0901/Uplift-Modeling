# Population and Sample Provenance

## Why This Exists

`README.md` describes the source file and the four samples drawn from it. Those
figures are read off this table rather than recomputed by hand, so a reader can
check any of them against `outputs/tables/sample_provenance.csv` instead of
taking them on trust.

## Protocol

- Population: `data/processed/criteo_indexed.parquet`, the indexed source with
  one row per row of the original CSV.
- Every rate is a plain average over the rows of the file named in that row.
- The visit effect is the treated rate minus the control rate. Randomization is
  what makes that difference an effect rather than a comparison, so no
  adjustment is applied and none is needed.
- `deviation_in_se` is each sample's distance from the population effect in
  units of that sample's own standard error.

## Measurements

| name                   | path                                                              | n        | treatment_rate | visit_rate | conversion_rate | control_visit_rate | treated_visit_rate | visit_effect_pp | standard_error_pp | deviation_in_se |
| ---------------------- | ----------------------------------------------------------------- | -------- | -------------- | ---------- | --------------- | ------------------ | ------------------ | --------------- | ----------------- | --------------- |
| Population             | D:/code/Vinsmart Future/data/processed/criteo_indexed.parquet     | 13979592 | 0.850000       | 0.046992   | 0.002917        | 3.820096           | 4.854336           | 1.034240        | 0.014632          | 0.000000        |
| Development            | D:/code/Vinsmart Future/data/processed/criteo_sample_500k.parquet | 500000   | 0.850104       | 0.047594   | 0.002946        | 3.942734           | 4.903400           | 0.960666        | 0.078424          | -0.938161       |
| Conversion development | D:/code/Vinsmart Future/data/processed/criteo_sample_2m.parquet   | 2000000  | 0.849934       | 0.047111   | 0.002914        | 3.754336           | 4.880029           | 1.125693        | 0.038432          | 2.379618        |
| Audit                  | D:/code/Vinsmart Future/data/processed/criteo_audit_1m.parquet    | 1000000  | 0.850017       | 0.047021   | 0.002919        | 3.781762           | 4.864491           | 1.082729        | 0.054503          | 0.889656        |
| Confirmatory           | D:/code/Vinsmart Future/data/processed/criteo_confirm_4m.parquet  | 4000000  | 0.850285       | 0.046880   | 0.002906        | 3.833283           | 4.838525           | 1.005241        | 0.027403          | -1.058225       |

## What This Shows

The source holds `13,979,592` rows, `85.00%`
of them treated, with a visit rate of `4.70%` and
a conversion rate of `0.29%`. Treating
everyone moves the visit rate from `3.820%` to
`4.854%`, an effect of
`+1.0342 pp`.

The samples are drawn by reservoir sampling on `row_id`, which does not look at
any column, so each should reproduce the population up to sampling noise. The
furthest is `Conversion development` at `+2.38` standard
errors. That is the check on the identity-based exclusion rule: excluding spent
rows by value instead would have shed the inert duplicate rows described in
`docs/determinism.md` and pushed the later samples' effects upward.

## Shared Rows

Overlap is counted by `row_id`, so two rows that happen to agree on every
column are still two rows. This is measured for every pair rather than for the
pairs a result happens to depend on.

| left                   | right                  | overlapping_rows | disjoint |
| ---------------------- | ---------------------- | ---------------- | -------- |
| Development            | Conversion development | 71193            | False    |
| Development            | Audit                  | 0                | True     |
| Development            | Confirmatory           | 0                | True     |
| Conversion development | Audit                  | 0                | True     |
| Conversion development | Confirmatory           | 0                | True     |
| Audit                  | Confirmatory           | 0                | True     |

Not every pair is disjoint: `Development` and `Conversion development` share `71,193` rows. Samples are only made disjoint from the samples they are drawn to avoid, and a pair that was never constrained overlaps at the rate two independent draws of that size would. This is worth stating plainly rather than rounding to "disjoint samples": any result that compares these two would be reusing rows, even though none of the results reported here does.

## Reproducible Outputs

- Measurements: `outputs/tables/sample_provenance.csv`
- Shared rows: `outputs/tables/sample_provenance_overlap.csv`
