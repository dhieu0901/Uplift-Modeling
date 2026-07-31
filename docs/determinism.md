# Reproducibility Notes

Every number in `README.md` should be recoverable from the commands in its
Reproduction section. Three details make that true rather than aspirational.

## 1. LightGBM is bit-exact in the thread count

`src/models/base.py` fits LightGBM with `n_jobs=-1`. Parallel histogram
construction is a common source of run-to-run drift, so this was measured rather
than assumed, on 400,000 Criteo audit rows with the project's own parameters:

| Comparison | Predictions identical | Max absolute difference |
|---|---|---:|
| `n_jobs=1` vs `n_jobs=-1` | yes (exact) | 0.0 |
| `n_jobs=-1` vs `n_jobs=-1` | yes (exact) | 0.0 |
| `n_jobs=1` vs `n_jobs=8` | yes (exact) | 0.0 |

Timing on 667,000 rows, 20 logical cores:

| Setting | Fit time |
|---|---:|
| `n_jobs=1` | 20.3 s |
| `n_jobs=-1` | 4.4 s |

The 4.6x speed-up is what makes the full evidence set, including the
four-million-row confirmatory test, rebuildable in an afternoon instead of a
weekend. Because the equality is exact, the results do not depend on how many
cores the reader happens to have.

Reproduce the check:

```python
import numpy as np
from lightgbm import LGBMClassifier
from src.data.criteo import load_criteo

dataset = load_criteo("data/processed/criteo_audit_1m.parquet", "visit")
X, y = dataset.X.iloc[:400_000], dataset.y.iloc[:400_000]


def fit(n_jobs: int) -> np.ndarray:
    model = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=n_jobs,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X, y)
    return model.predict_proba(X)[:, 1]


assert np.array_equal(fit(1), fit(-1))
```

## 2. Model seeds are derived from names, not positions

Seeds used to be `random_state + position_in_the_factory_dict`. That coupling is
a quiet reproducibility bug: adding one candidate shifts the seed of every
candidate after it, so a run before and a run after that change are not
comparable even for the learners both runs contain.

`_model_seed` in `src/experiments/honest_uplift.py` now hashes the model name, so
each seed is a function of the base seed and the model name only. Adding
`random_targeting` to the registry left every other learner's fit untouched.

This change is also why the internal locked test moved. See the note on seed
sensitivity in `README.md`: the same 200,000-row partition returned
`+168.5 [-53.0, 390.0]` under position-derived seeds and
`+266.5 [43.8, 489.2]` under name-derived seeds. Nothing about the data or the
protocol changed between those two runs. That spread is the reason the project
stopped treating a 200,000-row evaluation as sufficient.

## 3. Sampling is seeded in the database, and rows have identities

`prepare_criteo_index` materializes the source CSV once as Parquet with a
`row_id` column equal to the row's position in the file. The scan runs with
`threads = 1` and `preserve_insertion_order = true`, which is what makes
`row_number()` reproducible. Every sample is then drawn from that file with
DuckDB's `USING SAMPLE reservoir(n) REPEATABLE (seed)`, so a sample is a pure
function of `(indexed source, sample size, seed, exclusion set)`.

`count_overlapping_rows` re-verifies disjointness afterwards as a separate
query rather than trusting the construction.

### Why identity and not value

An earlier version excluded rows whose hash of *all columns* appeared in a
previous sample. That sounds equivalent and is not, because the source file
contains duplicate rows:

| Group | Rows | Treated | Control visit | Treated visit | Treatment effect |
|---|---:|---:|---:|---:|---:|
| Values duplicated elsewhere | 2,221,150 | 95.71% | 0.0000% | 0.0020% | +0.002 pp |
| Values unique | 11,758,442 | 82.98% | 4.0018% | 5.9116% | +1.910 pp |

The duplicated rows are almost entirely inert: no visits, no conversions, and
overwhelmingly in the treated arm. Hash exclusion drops *every* copy as soon as
one is drawn, so each successive sample shed more inert rows and its measured
treatment effect drifted upward:

| Sample | Exclusion | Effect under hash exclusion | Under identity exclusion |
|---|---|---:|---:|
| Population (13,979,592 rows) | — | +1.0342 pp | +1.0342 pp |
| 500,000 | none | +0.9607 pp | +0.9607 pp |
| 1,000,000 audit | 2.5M rows | +1.2241 pp (+3.5 SE) | +1.0827 pp (+0.9 SE) |
| 4,000,000 confirmatory | 3.5M rows | +1.3199 pp (+10.5 SE) | +1.0052 pp (-1.1 SE) |

The 500,000-row sample excludes nothing, so it was unaffected either way, and it
is what first showed the other two were drifting. Under identity exclusion every
sample is statistically indistinguishable from the population.

This was a real defect: it inflated the headline effect by roughly 28% before it
was found. `test_audit_sample_keeps_untouched_duplicates_of_used_rows` pins the
corrected behaviour.

## Known sources of variation

- Different LightGBM major versions can change tree construction. Pin with
  `requirements.lock.txt` to match the published numbers exactly.
- The sklearn fallback in `src/models/base.py` activates only when LightGBM is
  missing and will produce different numbers. CI installs LightGBM.
- `outputs/*/runtime` tables record wall-clock time and will differ per machine.
  Nothing else in `outputs/` should.
