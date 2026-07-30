from pathlib import Path

import pandas as pd
import pytest

from src.data.criteo import (
    CRITEO_FEATURE_COLUMNS,
    count_overlapping_rows,
    load_criteo,
    prepare_criteo_audit_sample,
)


def test_load_criteo_csv_selects_only_pretreatment_features(tmp_path: Path):
    row = {feature: float(index) for index, feature in enumerate(CRITEO_FEATURE_COLUMNS)}
    raw = pd.DataFrame(
        [
            {**row, "treatment": 1, "conversion": 0, "visit": 1, "exposure": 1},
            {**row, "treatment": 0, "conversion": 0, "visit": 0, "exposure": 0},
        ]
    )
    path = tmp_path / "criteo.csv"
    raw.to_csv(path, index=False)

    dataset = load_criteo(path, outcome="visit")

    assert list(dataset.X.columns) == CRITEO_FEATURE_COLUMNS
    assert "exposure" not in dataset.X.columns
    assert dataset.y.tolist() == [1, 0]
    assert dataset.treatment.tolist() == [1, 0]


def test_prepare_audit_sample_excludes_prior_rows(tmp_path: Path):
    rows = []
    for index in range(100):
        rows.append(
            {
                **{
                    feature: float(index + feature_index / 100)
                    for feature_index, feature in enumerate(
                        CRITEO_FEATURE_COLUMNS
                    )
                },
                "treatment": index % 2,
                "conversion": int(index % 20 == 0),
                "visit": int(index % 5 == 0),
                "exposure": int(index % 3 == 0),
            }
        )
    raw = pd.DataFrame(rows)
    raw_path = tmp_path / "raw.csv.gz"
    excluded_path = tmp_path / "excluded.parquet"
    output_path = tmp_path / "audit.parquet"
    raw.to_csv(raw_path, index=False, compression="gzip")
    raw.iloc[:20].to_parquet(excluded_path, index=False)

    prepare_criteo_audit_sample(
        raw_path,
        [excluded_path],
        output_path=output_path,
        sample_size=40,
        random_state=3,
    )
    audit = pd.read_parquet(output_path)

    assert len(audit) == 40
    assert set(audit["f0"]).isdisjoint(set(raw.iloc[:20]["f0"]))
    assert count_overlapping_rows(output_path, excluded_path) == 0


def test_overlap_count_finds_shared_rows(tmp_path: Path):
    frame = _benchmark_frame(30)
    left_path = tmp_path / "left.parquet"
    right_path = tmp_path / "right.parquet"
    frame.iloc[:20].to_parquet(left_path, index=False)
    frame.iloc[15:].to_parquet(right_path, index=False)

    # Rows 15-19 appear in both files.
    assert count_overlapping_rows(left_path, right_path) == 5
    assert count_overlapping_rows(right_path, left_path) == 5
    assert count_overlapping_rows(left_path, left_path) == 20


def test_overlap_count_requires_parquet(tmp_path: Path):
    parquet_path = tmp_path / "sample.parquet"
    csv_path = tmp_path / "sample.csv"
    frame = _benchmark_frame(5)
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Parquet"):
        count_overlapping_rows(parquet_path, csv_path)


def _benchmark_frame(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **{
                    feature: float(index + feature_index / 100)
                    for feature_index, feature in enumerate(CRITEO_FEATURE_COLUMNS)
                },
                "treatment": index % 2,
                "conversion": int(index % 20 == 0),
                "visit": int(index % 5 == 0),
                "exposure": int(index % 3 == 0),
            }
            for index in range(n_rows)
        ]
    )
