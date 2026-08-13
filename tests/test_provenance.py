from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.criteo import CRITEO_FEATURE_COLUMNS
from src.data.provenance import overlap_matrix, summarise_sample, summarise_samples


def _write_sample(
    path: Path,
    n_treated: int,
    visits_treated: int,
    n_control: int,
    visits_control: int,
) -> Path:
    n = n_treated + n_control
    frame = pd.DataFrame(
        {
            "row_id": range(n),
            "treatment": [1] * n_treated + [0] * n_control,
            "visit": (
                [1] * visits_treated
                + [0] * (n_treated - visits_treated)
                + [1] * visits_control
                + [0] * (n_control - visits_control)
            ),
            "conversion": [0] * n,
            "exposure": [0] * n,
        }
    )
    # count_overlapping_rows validates the full Criteo schema before querying,
    # so the fixture has to carry the feature columns even though none of the
    # measurements here read them.
    for column in CRITEO_FEATURE_COLUMNS:
        frame[column] = 0.0
    frame.to_parquet(path, index=False)
    return path


def test_the_effect_is_the_treated_rate_minus_the_control_rate(tmp_path):
    path = _write_sample(
        tmp_path / "sample.parquet",
        n_treated=800,
        visits_treated=80,
        n_control=200,
        visits_control=10,
    )

    summary = summarise_sample("Sample", path)

    assert summary.n == 1000
    assert summary.treatment_rate == pytest.approx(0.8)
    assert summary.treated_visit_rate == pytest.approx(10.0)
    assert summary.control_visit_rate == pytest.approx(5.0)
    assert summary.visit_effect_pp == pytest.approx(5.0)
    expected_se = 100.0 * np.sqrt(0.10 * 0.90 / 800 + 0.05 * 0.95 / 200)
    assert summary.standard_error_pp == pytest.approx(expected_se)


def test_distance_from_the_population_is_scaled_by_each_samples_own_noise(tmp_path):
    # Both samples miss the population effect by the same amount. Reporting the
    # raw gap would call them equally far off; the large one is the one that
    # cannot explain its gap by sampling noise.
    population = _write_sample(
        tmp_path / "population.parquet",
        n_treated=80_000,
        visits_treated=8_000,
        n_control=20_000,
        visits_control=1_000,
    )
    small = _write_sample(
        tmp_path / "small.parquet",
        n_treated=800,
        visits_treated=88,
        n_control=200,
        visits_control=10,
    )
    large = _write_sample(
        tmp_path / "large.parquet",
        n_treated=80_000,
        visits_treated=8_800,
        n_control=20_000,
        visits_control=1_000,
    )

    table = summarise_samples(population, {"Small": small, "Large": large}).set_index(
        "name"
    )

    gaps = table["visit_effect_pp"] - table.loc["Population", "visit_effect_pp"]
    assert gaps["Small"] == pytest.approx(gaps["Large"])
    assert abs(table.loc["Small", "deviation_in_se"]) < 1.0
    assert abs(table.loc["Large", "deviation_in_se"]) > 5.0


def test_overlap_is_counted_by_identity_not_by_matching_values(tmp_path):
    # Two samples with identical column values but disjoint row_ids describe
    # different users. A value-based check would call this a total overlap.
    left = _write_sample(
        tmp_path / "left.parquet",
        n_treated=80,
        visits_treated=8,
        n_control=20,
        visits_control=1,
    )
    right_frame = pd.read_parquet(left)
    right_frame["row_id"] = right_frame["row_id"] + 1000
    right = tmp_path / "right.parquet"
    right_frame.to_parquet(right, index=False)

    matrix = overlap_matrix({"Left": left, "Right": right, "Left again": left})

    by_pair = matrix.set_index(["left", "right"])
    assert by_pair.loc[("Left", "Right"), "overlapping_rows"] == 0
    assert bool(by_pair.loc[("Left", "Right"), "disjoint"])
    assert by_pair.loc[("Left", "Left again"), "overlapping_rows"] == 100
    assert not bool(by_pair.loc[("Left", "Left again"), "disjoint"])


def test_a_sample_with_one_empty_arm_is_refused(tmp_path):
    path = _write_sample(
        tmp_path / "treated_only.parquet",
        n_treated=100,
        visits_treated=10,
        n_control=0,
        visits_control=0,
    )

    with pytest.raises(ValueError, match="empty treatment arm"):
        summarise_sample("Treated only", path)
