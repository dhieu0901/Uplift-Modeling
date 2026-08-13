# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.data.provenance import overlap_matrix, summarise_samples
from src.reporting import dataframe_to_markdown

DEFAULT_SAMPLES = (
    "Development=data/processed/criteo_sample_500k.parquet,"
    "Conversion development=data/processed/criteo_sample_2m.parquet,"
    "Audit=data/processed/criteo_audit_1m.parquet,"
    "Confirmatory=data/processed/criteo_confirm_4m.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the population and every drawn sample, so the figures "
            "README.md quotes for them come from a tracked file rather than "
            "from a one-off query."
        )
    )
    parser.add_argument(
        "--population-path",
        default="data/processed/criteo_indexed.parquet",
    )
    parser.add_argument(
        "--samples",
        default=DEFAULT_SAMPLES,
        help="Comma-separated name=path pairs, in the order to report them.",
    )
    parser.add_argument("--report-path", default="outputs/sample_provenance.md")
    parser.add_argument(
        "--table-path",
        default="outputs/tables/sample_provenance.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = _parse_samples(args.samples)
    table = summarise_samples(ROOT / args.population_path, samples)
    overlaps = overlap_matrix(samples)

    report_path = ROOT / args.report_path
    table_path = ROOT / args.table_path
    overlap_path = table_path.with_name(f"{table_path.stem}_overlap.csv")
    for path in (report_path, table_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    overlaps.to_csv(overlap_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        build_report(args, table, overlaps, table_path, overlap_path),
        encoding="utf-8",
    )

    print(f"Provenance report: {report_path}")
    print(table.to_string(index=False))
    print(overlaps.to_string(index=False))


def _parse_samples(value: str) -> dict[str, Path]:
    samples: dict[str, Path] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Expected name=path, received {item!r}. Without the name the "
                "report cannot say which sample a row describes."
            )
        name, path = item.split("=", 1)
        samples[name.strip()] = ROOT / path.strip()
    if not samples:
        raise ValueError("At least one sample is required.")
    return samples


def _disjointness_verdict(overlaps) -> str:
    """Say what the overlap table shows, including when it is not all zeros."""
    shared = overlaps[~overlaps["disjoint"]]
    if shared.empty:
        return (
            "Every pair is disjoint, so no row contributes to two of these "
            "samples."
        )
    listed = "; ".join(
        f"`{row.left}` and `{row.right}` share "
        f"`{row.overlapping_rows:,}` rows"
        for row in shared.itertuples()
    )
    return (
        f"Not every pair is disjoint: {listed}. Samples are only made disjoint "
        "from the samples they are drawn to avoid, and a pair that was never "
        "constrained overlaps at the rate two independent draws of that size "
        "would. This is worth stating plainly rather than rounding to "
        "\"disjoint samples\": any result that compares these two would be "
        "reusing rows, even though none of the results reported here does."
    )


def build_report(
    args: argparse.Namespace,
    table,
    overlaps,
    table_path: Path,
    overlap_path: Path,
) -> str:
    population = table.iloc[0]
    drawn = table.iloc[1:]
    worst = drawn.loc[drawn["deviation_in_se"].abs().idxmax()]
    return f"""# Population and Sample Provenance

## Why This Exists

`README.md` describes the source file and the four samples drawn from it. Those
figures are read off this table rather than recomputed by hand, so a reader can
check any of them against `{table_path.relative_to(ROOT).as_posix()}` instead of
taking them on trust.

## Protocol

- Population: `{Path(args.population_path).as_posix()}`, the indexed source with
  one row per row of the original CSV.
- Every rate is a plain average over the rows of the file named in that row.
- The visit effect is the treated rate minus the control rate. Randomization is
  what makes that difference an effect rather than a comparison, so no
  adjustment is applied and none is needed.
- `deviation_in_se` is each sample's distance from the population effect in
  units of that sample's own standard error.

## Measurements

{dataframe_to_markdown(table)}

## What This Shows

The source holds `{population['n']:,}` rows, `{100.0 * population['treatment_rate']:.2f}%`
of them treated, with a visit rate of `{100.0 * population['visit_rate']:.2f}%` and
a conversion rate of `{100.0 * population['conversion_rate']:.2f}%`. Treating
everyone moves the visit rate from `{population['control_visit_rate']:.3f}%` to
`{population['treated_visit_rate']:.3f}%`, an effect of
`+{population['visit_effect_pp']:.4f} pp`.

The samples are drawn by reservoir sampling on `row_id`, which does not look at
any column, so each should reproduce the population up to sampling noise. The
furthest is `{worst['name']}` at `{worst['deviation_in_se']:+.2f}` standard
errors. That is the check on the identity-based exclusion rule: excluding spent
rows by value instead would have shed the inert duplicate rows described in
`docs/determinism.md` and pushed the later samples' effects upward.

## Shared Rows

Overlap is counted by `row_id`, so two rows that happen to agree on every
column are still two rows. This is measured for every pair rather than for the
pairs a result happens to depend on.

{dataframe_to_markdown(overlaps)}

{_disjointness_verdict(overlaps)}

## Reproducible Outputs

- Measurements: `{table_path.relative_to(ROOT).as_posix()}`
- Shared rows: `{overlap_path.relative_to(ROOT).as_posix()}`
"""


if __name__ == "__main__":
    main()
