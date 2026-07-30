import numpy as np
import pandas as pd
import pytest

from src.evaluation.uplift import (
    exact_qini_curve,
    exact_uplift_curve,
    incremental_outcome,
    qini_coefficient,
    separate_relative_auuc,
)


def test_incremental_outcome_uses_treated_control_difference():
    y = np.array([1, 1, 0, 0])
    treatment = np.array([1, 1, 0, 0])

    assert incremental_outcome(y, treatment) == 4.0


def test_qini_coefficient_is_zero_for_random_baseline_line():
    curve = pd.DataFrame(
        {
            "fraction": [0.0, 0.5, 1.0],
            "incremental_outcome": [0.0, 5.0, 10.0],
        }
    )

    assert qini_coefficient(curve) == 0.0


def test_separate_relative_auuc_matches_criteo_formula():
    y = np.array([1, 0, 1, 0])
    treatment = np.array([1, 1, 0, 0])
    score = np.array([0.9, 0.1, 0.2, 0.8])

    assert separate_relative_auuc(y, treatment, score) == 0.25


def test_exact_curves_match_scikit_uplift_with_tied_scores():
    sklift_metrics = pytest.importorskip("sklift.metrics")
    y = np.array([1, 0, 1, 0, 1, 0])
    treatment = np.array([1, 0, 1, 0, 1, 0])
    score = np.array([0.9, 0.9, 0.7, 0.5, 0.5, 0.1])

    expected_uplift_x, expected_uplift_y = sklift_metrics.uplift_curve(
        y, score, treatment
    )
    expected_qini_x, expected_qini_y = sklift_metrics.qini_curve(
        y, score, treatment
    )
    uplift_result = exact_uplift_curve(y, treatment, score)
    qini_result = exact_qini_curve(y, treatment, score)

    np.testing.assert_array_equal(uplift_result["n_targeted"], expected_uplift_x)
    np.testing.assert_allclose(
        uplift_result["incremental_outcome"], expected_uplift_y
    )
    np.testing.assert_array_equal(qini_result["n_targeted"], expected_qini_x)
    np.testing.assert_allclose(qini_result["qini_value"], expected_qini_y)


def test_exact_curve_rejects_non_binary_outcome():
    with pytest.raises(ValueError, match="y may contain only"):
        exact_uplift_curve([0, 2], [0, 1], [0.1, 0.2])
