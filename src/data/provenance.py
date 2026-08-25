from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.data.criteo import (
    _import_duckdb,
    _require_file,
    _sql_path,
    count_overlapping_rows,
)

_SUMMARY_SQL = """
SELECT
    count(*) AS n,
    avg(treatment) AS treatment_rate,
    avg(visit) AS visit_rate,
    avg(conversion) AS conversion_rate,
    sum(CASE WHEN treatment = 1 THEN 1 ELSE 0 END) AS n_treated,
    sum(CASE WHEN treatment = 1 THEN visit ELSE 0 END) AS visits_treated,
    sum(CASE WHEN treatment = 0 THEN 1 ELSE 0 END) AS n_control,
    sum(CASE WHEN treatment = 0 THEN visit ELSE 0 END) AS visits_control
FROM read_parquet('{path}')
"""


@dataclass(frozen=True)
class SampleSummary:
    """What a sample is, measured rather than assumed."""

    name: str
    path: str
    n: int
    treatment_rate: float
    visit_rate: float
    conversion_rate: float
    control_visit_rate: float
    treated_visit_rate: float
    visit_effect_pp: float
    standard_error_pp: float


#: The repository root, used to record sample paths relative to it.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _portable_path(path: str | Path) -> str:
    """Record a sample by where it sits in the repository, not on this disk.

    An absolute path is a property of the machine that ran the script, so
    writing one into a tracked evidence table guarantees that table can never
    reproduce and leaks the local directory layout for no benefit. A path
    outside the repository has no relative form and is kept as given.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(_REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def summarise_sample(name: str, path: str | Path) -> SampleSummary:
    """Measure size, assignment balance, outcome rates, and the visit effect.

    The aggregation runs in DuckDB rather than pandas because the indexed
    source is fourteen million rows and only eight numbers are needed from it.
    """
    resolved = _require_file(path)
    duckdb = _import_duckdb()
    with duckdb.connect() as connection:
        row = connection.execute(
            _SUMMARY_SQL.format(path=_sql_path(resolved))
        ).fetchone()
    if row is None:
        raise ValueError(f"{name} is empty, so there is nothing to summarise.")

    (
        n,
        treatment_rate,
        visit_rate,
        conversion_rate,
        n_treated,
        visits_treated,
        n_control,
        visits_control,
    ) = row
    if not n_treated or not n_control:
        raise ValueError(
            f"{name} has an empty treatment arm, so no effect is defined."
        )

    treated_rate = visits_treated / n_treated
    control_rate = visits_control / n_control
    return SampleSummary(
        name=name,
        path=_portable_path(path),
        n=int(n),
        treatment_rate=float(treatment_rate),
        visit_rate=float(visit_rate),
        conversion_rate=float(conversion_rate),
        control_visit_rate=100.0 * control_rate,
        treated_visit_rate=100.0 * treated_rate,
        visit_effect_pp=100.0 * (treated_rate - control_rate),
        standard_error_pp=100.0
        * _binary_difference_se(
            treated_rate, int(n_treated), control_rate, int(n_control)
        ),
    )


def summarise_samples(
    population_path: str | Path,
    samples: Mapping[str, str | Path],
) -> pd.DataFrame:
    """Summarise the population and each drawn sample in one table.

    Each sample's distance from the population is reported in units of that
    sample's own standard error. A sample is meant to be a smaller copy of the
    population, so the check has to be against sampling noise rather than
    against a fixed tolerance: the four samples differ in size by a factor of
    eight and a gap that is unremarkable in the smallest would be a problem in
    the largest.
    """
    population = summarise_sample("Population", population_path)
    rows = [{**asdict(population), "deviation_in_se": 0.0}]
    for name, path in samples.items():
        summary = summarise_sample(name, path)
        rows.append(
            {
                **asdict(summary),
                "deviation_in_se": (
                    summary.visit_effect_pp - population.visit_effect_pp
                )
                / summary.standard_error_pp,
            }
        )
    return pd.DataFrame(rows)


def overlap_matrix(samples: Mapping[str, str | Path]) -> pd.DataFrame:
    """Count shared rows for every pair of samples, not just the pairs in use.

    Disjointness was previously verified only where a result depended on it,
    which leaves the pairs nobody thought to check unmeasured. Every pair is
    cheap here, and a pair that turns out to overlap is worth knowing about
    before someone builds on the assumption that it does not.
    """
    names = list(samples)
    rows = []
    for left, right in combinations(names, 2):
        shared = count_overlapping_rows(samples[left], samples[right])
        rows.append(
            {
                "left": left,
                "right": right,
                "overlapping_rows": shared,
                "disjoint": shared == 0,
            }
        )
    return pd.DataFrame(rows)


def _binary_difference_se(
    left_rate: float,
    left_n: int,
    right_rate: float,
    right_n: int,
) -> float:
    variance = (
        left_rate * (1.0 - left_rate) / left_n
        + right_rate * (1.0 - right_rate) / right_n
    )
    return float(variance**0.5)
