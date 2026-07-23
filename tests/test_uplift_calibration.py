import numpy as np
import pandas as pd

from src.evaluation.calibration import (
    summarize_uplift_calibration,
    uplift_calibration_table,
)
from src.models.uplift_calibration import UpliftIsotonicCalibrator


def test_isotonic_calibrator_returns_monotonic_finite_predictions():
    score = np.linspace(-0.1, 0.2, 200)
    treatment = np.tile([0, 1], 100)
    y = ((score > 0.02) & (treatment == 1)).astype(int)

    calibrator = UpliftIsotonicCalibrator().fit(
        score, y, treatment, propensity=0.5
    )
    prediction = calibrator.predict(score)

    assert calibrator.propensity_ == 0.5
    assert calibrator.n_bins_ == 100
    assert np.isfinite(prediction).all()
    assert (np.diff(prediction) >= 0).all()
    assert ((prediction >= -1.0) & (prediction <= 1.0)).all()


def test_uplift_calibration_table_has_equal_sized_bins():
    y = np.tile([0, 1, 0, 1], 25)
    treatment = np.tile([0, 0, 1, 1], 25)
    score = np.linspace(0.2, -0.1, 100)

    result = uplift_calibration_table(y, treatment, score, n_bins=5)

    assert result["n"].tolist() == [20, 20, 20, 20, 20]
    assert result["bin"].tolist() == [1, 2, 3, 4, 5]
    assert result["predicted_uplift"].is_monotonic_decreasing


def test_calibration_summary_is_ideal_when_prediction_matches_observation():
    table = pd.DataFrame(
        {
            "n": [100, 100, 100],
            "predicted_uplift": [-0.1, 0.0, 0.1],
            "observed_uplift": [-0.1, 0.0, 0.1],
        }
    )

    result = summarize_uplift_calibration(table)

    assert np.isclose(result["weighted_bias"], 0.0)
    assert np.isclose(result["weighted_mae"], 0.0)
    assert np.isclose(result["weighted_rmse"], 0.0)
    assert np.isclose(result["calibration_intercept"], 0.0)
    assert np.isclose(result["calibration_slope"], 1.0)
