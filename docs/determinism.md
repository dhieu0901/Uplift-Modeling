# Reproducibility Notes

Every number in `README.md` is recoverable from the commands in its Reproduction
section. Three details make that true rather than aspirational.

## Sampling is seeded in the database, and rows have identities

`prepare_criteo_index` materializes the source CSV once as Parquet with a
`row_id` equal to the row's position in the file. That scan runs with
`threads = 1` and `preserve_insertion_order = true`, which is what makes
`row_number()` reproducible. Every sample is then drawn from that file with
DuckDB's `USING SAMPLE reservoir(n) REPEATABLE (seed)`, so a sample is a pure
function of `(indexed source, size, seed, exclusion set)`.
`count_overlapping_rows` re-verifies disjointness afterwards as a separate query
rather than trusting the construction.

### Why identity and not value

Samples are made disjoint by `row_id`, not by a hash of the row's columns. The
two are not equivalent here, because the source contains duplicate rows and they
are far from neutral:

| Group | Rows | Treated | Treatment effect |
|---|---:|---:|---:|
| Values duplicated elsewhere | 2,221,150 | 95.71% | +0.002 pp |
| Values unique | 11,758,442 | 82.98% | +1.910 pp |

The duplicated rows are inert: no visits, no conversions, overwhelmingly treated.
Excluding by value would drop *every* copy as soon as one is drawn, so each
successive sample would shed more inert rows and its measured treatment effect
would drift above the population's +1.0342 pp. Excluding by identity removes
exactly the rows already spent, and all four samples land within about 1 SE of
the population.

`test_audit_sample_keeps_untouched_duplicates_of_used_rows` pins this behaviour.

## Model seeds are derived from names, not positions

`_model_seed` hashes the model name, so each seed is a function of the base seed
and the model name only. Deriving seeds from a model's position in the candidate
list instead would mean that adding one candidate reseeds every candidate after
it, and two runs sharing a learner would not be comparable.

Seed choice is not a cosmetic detail at this sample size. On the same 200,000-row
partition, two seeding conventions returned `+168.5 [-53.0, 390.0]` and
`+266.5 [43.8, 489.2]` - one spanning zero and one not - with nothing about the
data or the protocol different between them. A 200,000-row evaluation cannot
resolve an effect this size, which is why the four-million-row confirmatory
sample exists.

## LightGBM is deterministic in the thread count

`src/models/base.py` fits with `n_jobs=-1`. Parallel histogram construction is a
common source of run-to-run drift, so this was measured rather than assumed: on
400,000 audit rows with the project's own parameters, predictions are
bit-identical across `n_jobs` of 1, 8, and -1. Because the equality is exact,
results do not depend on how many cores the reader has, and the 4.6x speed-up is
free.

## Random forests are not, and the base learner comparison measures it

`scripts/run_base_learner_comparison.py` runs the same protocol on three base
learner families, so it also serves as a reproducibility test of each one. Three
independent executions on the same 1,000,000-row sample, same seed, same three
folds:

| Base learner family | Largest spread across three runs | Ranks |
|---|---:|---|
| `gradient_boosting` | 0 | identical |
| `linear` | 0 | identical |
| `forest` | 37.8 incremental outcomes | identical |

The two boosted and linear columns are identical to the last bit. The forest
column is not. scikit-learn's forests average their trees by accumulating
predictions into a shared array as worker threads finish, and floating point
addition is not associative, so the order the threads happen to finish in
changes the last bits of every score.

The spread is 37.8 against a selection half-width near 441, so it is small
relative to what the sample can resolve. It is also small relative to the gaps
between users: all twenty-one selection ranks were identical in all three runs
and every family picked the same champion each time. The estimate drifts, the
decision does not.

One derived quantity does move visibly. `margin_over_halfwidth` for the forest
came out 0.28, 0.37, and 0.31 across the three runs, because it divides a small
gap by a stable half-width and inherits the drift of both endpoints. It stays
far below 1 in every run, which is the reading that matters, but it should be
quoted as a range rather than as a figure.

Passing `n_jobs=1` to the forest would remove the drift at a large cost in
runtime. This project measured the drift instead and reports it, because the
forest is a comparison column rather than a production model: no locked policy
and no published headline depends on it.

## Known sources of variation

- Different LightGBM major versions can change tree construction. Pin with
  `requirements.lock.txt` to match the published numbers.
- The sklearn fallback in `src/models/base.py` activates only when LightGBM is
  missing and produces different numbers. CI installs LightGBM.
- `outputs/*/runtime` tables record wall-clock time and differ per machine.
  Nothing else in `outputs/` should.
