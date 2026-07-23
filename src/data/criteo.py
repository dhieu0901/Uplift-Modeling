from __future__ import annotations

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


@dataclass(frozen=True)
class CriteoDataset:
    """Criteo dataset prepared for the binary-treatment problem."""

    X: pd.DataFrame
    y: pd.Series
    treatment: pd.Series
    raw: pd.DataFrame
    feature_columns: list[str]
    outcome: str


@dataclass(frozen=True)
class CriteoSummary:
    """Dataset-level and treatment-arm statistics."""

    overall: pd.DataFrame
    by_treatment: pd.DataFrame


def prepare_criteo_sample(
    raw_path: str | Path = "data/criteo-uplift-v2.1.csv.gz",
    output_path: str | Path = "data/processed/criteo_sample_500k.parquet",
    sample_size: int = 500_000,
    random_state: int = 42,
    force: bool = False,
) -> Path:
    """Create a reproducible reservoir sample from compressed Criteo data and save it as Parquet."""
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero.")

    raw_path = _require_file(raw_path)
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duckdb = _import_duckdb()
    raw_sql = _sql_path(raw_path)
    output_sql = _sql_path(output_path)

    with duckdb.connect() as connection:
        columns = _read_csv_columns(connection, raw_sql)
        _validate_columns(columns)
        connection.execute(
            f"""
            COPY (
                SELECT {", ".join(CRITEO_REQUIRED_COLUMNS)}
                FROM read_csv_auto('{raw_sql}', header = true, compression = 'gzip')
                USING SAMPLE reservoir({int(sample_size)} ROWS) REPEATABLE ({int(random_state)})
            ) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

    return output_path


def summarize_criteo(
    raw_path: str | Path = "data/criteo-uplift-v2.1.csv.gz",
) -> CriteoSummary:
    """Scan the full Criteo dataset once to calculate outcome and treatment rates."""
    raw_path = _require_file(raw_path)
    duckdb = _import_duckdb()
    raw_sql = _sql_path(raw_path)

    with duckdb.connect() as connection:
        columns = _read_csv_columns(connection, raw_sql)
        _validate_columns(columns)
        by_treatment = connection.execute(
            f"""
            SELECT
                treatment,
                count(*) AS n,
                avg(visit) AS visit_rate,
                avg(conversion) AS conversion_rate,
                avg(exposure) AS exposure_rate,
                sum(visit) AS visits,
                sum(conversion) AS conversions
            FROM read_csv_auto('{raw_sql}', header = true, compression = 'gzip')
            GROUP BY treatment
            ORDER BY treatment
            """
        ).fetch_df()

    n = int(by_treatment["n"].sum())
    treated_n = int(by_treatment.loc[by_treatment["treatment"] == 1, "n"].sum())
    overall = pd.DataFrame(
        [
            {
                "n": n,
                "treatment_rate": treated_n / n,
                "visit_rate": by_treatment["visits"].sum() / n,
                "conversion_rate": by_treatment["conversions"].sum() / n,
                "exposure_rate": (
                    by_treatment["exposure_rate"] * by_treatment["n"]
                ).sum()
                / n,
                "visits": int(by_treatment["visits"].sum()),
                "conversions": int(by_treatment["conversions"].sum()),
            }
        ]
    )

    return CriteoSummary(overall=overall, by_treatment=by_treatment)


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


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize distributions and missing rates for the 12 anonymized features."""
    summary = df[CRITEO_FEATURE_COLUMNS].describe(percentiles=[0.01, 0.5, 0.99]).T
    summary["missing_rate"] = df[CRITEO_FEATURE_COLUMNS].isna().mean()
    return summary[["count", "mean", "std", "min", "1%", "50%", "99%", "max", "missing_rate"]]


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
