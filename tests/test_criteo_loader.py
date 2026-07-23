from pathlib import Path

import pandas as pd

from src.data.criteo import CRITEO_FEATURE_COLUMNS, feature_summary, load_criteo


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


def test_feature_summary_contains_missing_rate():
    raw = pd.DataFrame({feature: [0.0, 1.0] for feature in CRITEO_FEATURE_COLUMNS})

    summary = feature_summary(raw)

    assert summary.shape[0] == len(CRITEO_FEATURE_COLUMNS)
    assert (summary["missing_rate"] == 0.0).all()
