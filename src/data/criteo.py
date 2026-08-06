from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


CRITEO_FEATURE_COLUMNS = [f"f{i}" for i in range(12)]
CRITEO_REQUIRED_COLUMNS = [
    *CRITEO_FEATURE_COLUMNS,
    "treatment",
    "conversion",
    "visit",
    "exposure",
]
CRITEO_OUTCOME_COLUMNS = ["visit", "conversion"]

#: Position of a row in the source file. Every sample carries it so that samples
#: can be made disjoint by identity instead of by value.
#:
#: Excluding by a hash of the row's values looks equivalent but is not. The
#: source file holds 2,221,150 rows whose values duplicate another row, and
#: those duplicates are almost entirely non-responders (a 0.002pp treatment
#: effect, against 1.910pp among unique rows). A value rule drops *every* copy
#: as soon as one is drawn, so later samples lose those inert rows and their
#: measured treatment effect reads high. Identity removes exactly the rows
#: already spent.
CRITEO_ROW_ID = "row_id"


@dataclass(frozen=True)
class CriteoDataset:
    """Criteo dataset prepared for the binary-treatment problem."""

    X: pd.DataFrame
    y: pd.Series
    treatment: pd.Series
    raw: pd.DataFrame
    feature_columns: list[str]
    outcome: str


def prepare_criteo_index(
    raw_path: str | Path = "data/criteo-uplift-v2.1.csv.gz",
    output_path: str | Path = "data/processed/criteo_indexed.parquet",
    force: bool = False,
) -> Path:
    """Materialize the source file once with a stable ``row_id`` per row.

    Every later sample is drawn from this file, so a row's identity is its
    position in the original CSV. The scan runs single-threaded with insertion
    order preserved, which is what makes ``row_number()`` reproducible.
    """
    raw_path = _require_file(raw_path)
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duckdb = _import_duckdb()
    raw_sql = _sql_path(raw_path)
    output_sql = _sql_path(output_path)

    with duckdb.connect() as connection:
        connection.execute("SET threads TO 1")
        connection.execute("SET preserve_insertion_order TO true")
        columns = _read_csv_columns(connection, raw_sql)
        _validate_columns(columns)
        connection.execute(
            f"""
            COPY (
                SELECT
                    (row_number() OVER () - 1) AS {CRITEO_ROW_ID},
                    {", ".join(CRITEO_REQUIRED_COLUMNS)}
                FROM read_csv_auto('{raw_sql}', header = true, compression = 'gzip')
            ) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    return output_path


def prepare_criteo_sample(
    index_path: str | Path = "data/processed/criteo_indexed.parquet",
    output_path: str | Path = "data/processed/criteo_sample_500k.parquet",
    sample_size: int = 500_000,
    random_state: int = 42,
    force: bool = False,
) -> Path:
    """Draw a reproducible reservoir sample from the indexed source file."""
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero.")

    index_path = _require_indexed_parquet(index_path)
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duckdb = _import_duckdb()

    with duckdb.connect() as connection:
        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{_sql_path(index_path)}')
                USING SAMPLE reservoir({int(sample_size)} ROWS)
                REPEATABLE ({int(random_state)})
            ) TO '{_sql_path(output_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    return output_path


def prepare_criteo_audit_sample(
    index_path: str | Path,
    excluded_sample_paths: Sequence[str | Path],
    output_path: str | Path = "data/processed/criteo_audit_1m.parquet",
    sample_size: int = 1_000_000,
    random_state: int = 777,
    force: bool = False,
) -> Path:
    """Draw a sample disjoint from prior samples by row identity.

    Rows are excluded by ``row_id``, so exactly the rows already spent are
    removed. Excluding by value hash instead would also remove every untouched
    row that happens to duplicate a used one, and in this dataset those
    duplicates are overwhelmingly inert -- see :data:`CRITEO_ROW_ID`.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero.")
    if not excluded_sample_paths:
        raise ValueError("At least one excluded sample path is required.")

    index_path = _require_indexed_parquet(index_path)
    excluded_paths = [_require_indexed_parquet(path) for path in excluded_sample_paths]
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duckdb = _import_duckdb()
    excluded_sql = " UNION ".join(
        f"SELECT {CRITEO_ROW_ID} FROM read_parquet('{_sql_path(path)}')"
        for path in excluded_paths
    )

    with duckdb.connect() as connection:
        connection.execute(
            f"""
            COPY (
                WITH excluded_ids AS (
                    {excluded_sql}
                )
                SELECT source.*
                FROM read_parquet('{_sql_path(index_path)}') AS source
                ANTI JOIN excluded_ids USING ({CRITEO_ROW_ID})
                USING SAMPLE reservoir({int(sample_size)} ROWS)
                REPEATABLE ({int(random_state)})
            ) TO '{_sql_path(output_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    return output_path


def count_overlapping_rows(
    left_path: str | Path,
    right_path: str | Path,
) -> int:
    """Count rows of ``left_path`` whose ``row_id`` also occurs in ``right_path``.

    Disjointness is the load-bearing assumption behind every confirmatory
    number in this project, so it is verified as a reusable query rather than
    trusted from how a sample was constructed. Identity, not value: two rows
    that merely share every column are still two different rows.
    """
    left = _require_indexed_parquet(left_path)
    right = _require_indexed_parquet(right_path)

    duckdb = _import_duckdb()
    with duckdb.connect() as connection:
        overlap = connection.execute(
            f"""
            SELECT count(*)
            FROM read_parquet('{_sql_path(left)}') AS left_rows
            SEMI JOIN (
                SELECT DISTINCT {CRITEO_ROW_ID}
                FROM read_parquet('{_sql_path(right)}')
            ) AS right_rows
            USING ({CRITEO_ROW_ID})
            """
        ).fetchone()[0]
    return int(overlap)


def load_criteo(
    path: str | Path = "data/processed/criteo_sample_500k.parquet",
    outcome: str = "visit",
) -> CriteoDataset:
    """Read a Criteo sample and separate features, treatment, and outcome."""
    if outcome not in CRITEO_OUTCOME_COLUMNS:
        raise ValueError(f"Invalid outcome: {outcome}")

    path = _require_file(path)
    if path.suffix.lower() == ".parquet":
        duckdb = _import_duckdb()
        with duckdb.connect() as connection:
            raw = connection.execute(
                f"SELECT * FROM read_parquet('{_sql_path(path)}')"
            ).fetch_df()
    else:
        raw = pd.read_csv(path)

    _validate_columns(raw.columns)
    X = raw[CRITEO_FEATURE_COLUMNS].astype("float32")
    y = raw[outcome].astype("int8")
    treatment = raw["treatment"].astype("int8")

    return CriteoDataset(
        X=X,
        y=y,
        treatment=treatment,
        raw=raw,
        feature_columns=CRITEO_FEATURE_COLUMNS.copy(),
        outcome=outcome,
    )


def subsample_criteo(
    dataset: CriteoDataset,
    max_rows: int | None,
    random_state: int,
) -> CriteoDataset:
    """Limit the row count while preserving treatment/outcome distributions."""
    if max_rows is None or max_rows >= len(dataset.X):
        return dataset
    if max_rows <= 0:
        raise ValueError("max_rows must be greater than zero.")

    indices = np.arange(len(dataset.X))
    strata = dataset.treatment.astype(str) + "_" + dataset.y.astype(str)
    selected, _ = train_test_split(
        indices,
        train_size=max_rows,
        random_state=random_state,
        stratify=strata,
    )
    return CriteoDataset(
        X=dataset.X.iloc[selected].reset_index(drop=True),
        y=dataset.y.iloc[selected].reset_index(drop=True),
        treatment=dataset.treatment.iloc[selected].reset_index(drop=True),
        raw=dataset.raw.iloc[selected].reset_index(drop=True),
        feature_columns=dataset.feature_columns,
        outcome=dataset.outcome,
    )


def _read_csv_columns(connection, raw_sql: str) -> list[str]:
    description = connection.execute(
        f"DESCRIBE SELECT * FROM read_csv_auto('{raw_sql}', header = true, compression = 'gzip')"
    ).fetch_df()
    return description["column_name"].tolist()


def _validate_columns(columns) -> None:
    missing = sorted(set(CRITEO_REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"Criteo dataset is missing required columns: {missing}")


def _require_indexed_parquet(path: str | Path) -> Path:
    """Resolve a Parquet sample and confirm it carries the row identifier."""
    resolved = _require_file(path)
    if resolved.suffix.lower() != ".parquet":
        raise ValueError(f"Expected a Parquet sample, received: {resolved}")

    duckdb = _import_duckdb()
    with duckdb.connect() as connection:
        columns = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{_sql_path(resolved)}')"
        ).fetch_df()["column_name"].tolist()
    if CRITEO_ROW_ID not in columns:
        raise ValueError(
            f"{resolved} has no '{CRITEO_ROW_ID}' column. Rebuild the samples "
            "with scripts/prepare_criteo.py so that rows can be excluded by "
            "identity instead of by value."
        )
    _validate_columns(columns)
    return resolved


def _require_file(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Data not found at: {resolved}")
    return resolved


def _import_duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "DuckDB is required to process Criteo data: python -m pip install duckdb"
        ) from exc
    return duckdb


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")
