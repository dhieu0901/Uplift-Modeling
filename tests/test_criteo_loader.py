from pathlib import Path

import pandas as pd

from src.data.criteo import (
    CRITEO_FEATURE_COLUMNS,
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
