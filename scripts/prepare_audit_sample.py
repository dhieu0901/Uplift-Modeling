# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.criteo import (
    count_overlapping_rows,
    prepare_criteo_audit_sample,
)
from src.reporting import dataframe_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a confirmatory Criteo sample after excluding every row "
            "hash used by prior development samples."
        )
    )
    parser.add_argument(
        "--index-path",
        default="data/processed/criteo_indexed.parquet",
        help="Indexed source file produced by scripts/prepare_criteo.py.",
    )
    parser.add_argument(
        "--excluded-paths",
        default=(
            "data/processed/criteo_sample_500k.parquet,"
            "data/processed/criteo_sample_2m.parquet"
        ),
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/criteo_audit_1m.parquet",
    )
    parser.add_argument("--sample-size", type=int, default=1_000_000)
    parser.add_argument("--random-state", type=int, default=777)
    parser.add_argument(
        "--report-path",
        default="outputs/audit_sample.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded_paths = [
        ROOT / item.strip()
        for item in args.excluded_paths.split(",")
        if item.strip()
    ]
    output_path = prepare_criteo_audit_sample(
        ROOT / args.index_path,
        excluded_paths,
        output_path=ROOT / args.output_path,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    audit_summary, overlap_summary = summarize_audit(
        output_path,
        excluded_paths,
    )
    report_path = ROOT / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(
            args,
            audit_summary,
            overlap_summary,
            output_path,
        ),
        encoding="utf-8",
    )
    print(f"Audit sample: {output_path}")
    print(overlap_summary.to_string(index=False))


def summarize_audit(
    audit_path: Path,
    excluded_paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_sql = _sql_path(audit_path)
    with duckdb.connect() as connection:
        audit_summary = connection.execute(
            f"""
            SELECT
                count(*) AS n,
                avg(treatment) AS treatment_rate,
                avg(visit) AS visit_rate,
                avg(conversion) AS conversion_rate
            FROM read_parquet('{audit_sql}')
            """
        ).fetch_df()
    rows = [
        {
            "excluded_sample": excluded_path.relative_to(ROOT).as_posix(),
            "overlap_rows": count_overlapping_rows(audit_path, excluded_path),
        }
        for excluded_path in excluded_paths
    ]
    return audit_summary, pd.DataFrame(rows)


def build_report(
    args,
    audit_summary: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    output_path: Path,
) -> str:
    return f"""# Confirmatory Criteo Audit Sample

## Construction

- Indexed source: `{args.index_path}`.
- Excluded development samples: `{args.excluded_paths}`.
- Requested rows: `{args.sample_size:,}`.
- Reservoir seed: `{args.random_state}`.
- Rows already used by an excluded sample were removed by `row_id` before
  reservoir sampling, so exactly those rows are withheld and untouched
  duplicates of them remain eligible.

## Audit Summary

{dataframe_to_markdown(audit_summary)}

## Disjointness Check

{dataframe_to_markdown(overlap_summary)}

The audit sample is stored at `{output_path.relative_to(ROOT).as_posix()}`.
Duplicate-valued rows sharing an excluded hash are conservatively removed.
"""


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


if __name__ == "__main__":
    main()
