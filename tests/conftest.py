from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.criteo import CRITEO_FEATURE_COLUMNS
from src.data.semisynthetic import (
    SemiSyntheticUpliftDataset,
    generate_semisynthetic_uplift,
)


@pytest.fixture
def small_uplift_dataset() -> SemiSyntheticUpliftDataset:
    """A small randomized dataset with a known CATE, for fast integration tests.

    The outcome rate is deliberately far higher than Criteo's so that a few
    thousand rows still populate every treatment/outcome stratum.
    """
    generator = np.random.default_rng(7)
    features = pd.DataFrame(
        generator.normal(size=(3000, len(CRITEO_FEATURE_COLUMNS))),
        columns=CRITEO_FEATURE_COLUMNS,
    )
    return generate_semisynthetic_uplift(
        features,
        control_rate=0.25,
        average_effect=0.05,
        heterogeneity_scale=0.10,
        treatment_propensity=0.5,
        random_state=11,
    )
