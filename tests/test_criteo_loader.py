from pathlib import Path

import pandas as pd
import pytest

from src.data.criteo import (
    CRITEO_FEATURE_COLUMNS,
    CRITEO_ROW_ID,
    count_overlapping_rows,
    load_criteo,
    prepare_criteo_audit_sample,
    prepare_criteo_index,
    prepare_criteo_sample,
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


def test_index_numbers_every_source_row_exactly_once(tmp_path: Path):
    raw_path = _write_source(tmp_path, _benchmark_frame(50))

    index_path = prepare_criteo_index(raw_path, tmp_path / "indexed.parquet")
    indexed = pd.read_parquet(index_path)

    assert len(indexed) == 50
    assert indexed[CRITEO_ROW_ID].tolist() == list(range(50))


def test_audit_sample_excludes_prior_rows_by_identity(tmp_path: Path):
    frame = _benchmark_frame(100)
    raw_path = _write_source(tmp_path, frame)
    index_path = prepare_criteo_index(raw_path, tmp_path / "indexed.parquet")
    excluded_path = prepare_criteo_sample(
        index_path,
        tmp_path / "excluded.parquet",
        sample_size=20,
        random_state=1,
    )

    audit_path = prepare_criteo_audit_sample(
        index_path,
        [excluded_path],
        output_path=tmp_path / "audit.parquet",
        sample_size=40,
        random_state=3,
    )
    audit = pd.read_parquet(audit_path)
    excluded = pd.read_parquet(excluded_path)

    assert len(audit) == 40
    assert set(audit[CRITEO_ROW_ID]).isdisjoint(set(excluded[CRITEO_ROW_ID]))
    assert count_overlapping_rows(audit_path, excluded_path) == 0


def test_audit_sample_keeps_untouched_duplicates_of_used_rows(tmp_path: Path):
    """A row is spent by being drawn, not by resembling something drawn.

    The source file duplicates every record, so excluding by value would drop
    both copies as soon as one is used. In the real dataset those duplicates
    are overwhelmingly non-responders, and removing them inflated the measured
    treatment effect by 28%. This pins the identity-based behaviour.
    """
    frame = pd.concat([_benchmark_frame(50), _benchmark_frame(50)], ignore_index=True)
    raw_path = _write_source(tmp_path, frame)
    index_path = prepare_criteo_index(raw_path, tmp_path / "indexed.parquet")
    excluded_path = prepare_criteo_sample(
        index_path,
        tmp_path / "excluded.parquet",
        sample_size=50,
        random_state=7,
    )

    audit_path = prepare_criteo_audit_sample(
        index_path,
        [excluded_path],
        output_path=tmp_path / "audit.parquet",
        sample_size=50,
        random_state=11,
    )
    audit = pd.read_parquet(audit_path)
    excluded = pd.read_parquet(excluded_path)

    # Every remaining row is available, so the audit takes all 50 of them.
    assert len(audit) == 50
    assert count_overlapping_rows(audit_path, excluded_path) == 0
    # The two samples together rebuild the source exactly: nothing was lost to
    # a value collision.
    assert set(audit[CRITEO_ROW_ID]) | set(excluded[CRITEO_ROW_ID]) == set(range(100))
    # And the shared feature values really do span both samples.
    assert not set(audit["f0"]).isdisjoint(set(excluded["f0"]))


def test_overlap_count_finds_shared_rows(tmp_path: Path):
    frame = _indexed_frame(30)
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
    frame = _indexed_frame(5)
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Parquet"):
        count_overlapping_rows(parquet_path, csv_path)


def test_overlap_count_rejects_a_sample_without_row_ids(tmp_path: Path):
    path = tmp_path / "legacy.parquet"
    _benchmark_frame(5).to_parquet(path, index=False)

    with pytest.raises(ValueError, match=CRITEO_ROW_ID):
        count_overlapping_rows(path, path)


def _write_source(tmp_path: Path, frame: pd.DataFrame) -> Path:
    raw_path = tmp_path / "raw.csv.gz"
    frame.to_csv(raw_path, index=False, compression="gzip")
    return raw_path


def _indexed_frame(n_rows: int) -> pd.DataFrame:
    frame = _benchmark_frame(n_rows)
    frame.insert(0, CRITEO_ROW_ID, range(n_rows))
    return frame


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
