import numpy as np
import pandas as pd

from src.data.criteo import CRITEO_FEATURE_COLUMNS, CriteoDataset
from src.experiments.splitting import split_train_validation_test


def test_honest_split_has_expected_sizes_and_preserves_strata():
    n = 200
    raw = pd.DataFrame(
        {
            **{
                feature: np.arange(n, dtype=float)
                for feature in CRITEO_FEATURE_COLUMNS
            },
            "treatment": np.tile([0, 0, 1, 1], n // 4),
            "visit": np.tile([0, 1, 0, 1], n // 4),
            "conversion": np.zeros(n, dtype=int),
            "exposure": np.zeros(n, dtype=int),
        }
    )
    dataset = CriteoDataset(
        X=raw[CRITEO_FEATURE_COLUMNS],
        y=raw["visit"],
        treatment=raw["treatment"],
        raw=raw,
        feature_columns=CRITEO_FEATURE_COLUMNS.copy(),
        outcome="visit",
    )

    splits = split_train_validation_test(dataset, random_state=7)

    assert len(splits.train.X) == 120
    assert len(splits.validation.X) == 40
    assert len(splits.test.X) == 40
    for split in (splits.train, splits.validation, splits.test):
        strata = split.treatment.astype(str) + "_" + split.y.astype(str)
        assert set(strata) == {"0_0", "0_1", "1_0", "1_1"}
