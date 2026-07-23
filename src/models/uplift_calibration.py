from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class UpliftIsotonicCalibrator:
    """Calibrate uplift scores using transformed outcomes on a separate set."""

    n_bins: int = 100
    model: IsotonicRegression | None = None
    propensity_: float | None = None
    n_bins_: int | None = None

    def fit(
        self,
        score,
        y,
        treatment,
        propensity: float | None = None,
    ) -> "UpliftIsotonicCalibrator":
        score_arr, y_arr, treatment_arr = _validate_inputs(score, y, treatment)
        propensity_value = (
            float(treatment_arr.mean()) if propensity is None else float(propensity)
        )
        if not 0.0 < propensity_value < 1.0:
            raise ValueError("propensity must be in the interval (0, 1).")
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2.")

        pseudo_outcome = (
            y_arr
            * (treatment_arr - propensity_value)
            / (propensity_value * (1.0 - propensity_value))
        )
        order = np.argsort(score_arr, kind="mergesort")
        score_sorted = score_arr[order]
        pseudo_sorted = pseudo_outcome[order]
        actual_bins = min(self.n_bins, len(score_arr))
        bin_ids = np.floor(
            np.arange(len(score_arr), dtype=float) * actual_bins / len(score_arr)
        ).astype(int)
        grouped_score = np.array(
            [score_sorted[bin_ids == bin_id].mean() for bin_id in range(actual_bins)]
        )
        grouped_pseudo_outcome = np.array(
            [pseudo_sorted[bin_ids == bin_id].mean() for bin_id in range(actual_bins)]
        )
        grouped_weight = np.bincount(bin_ids, minlength=actual_bins)

        self.model = IsotonicRegression(
            y_min=-1.0,
            y_max=1.0,
            out_of_bounds="clip",
        )
        self.model.fit(
            grouped_score,
            grouped_pseudo_outcome,
            sample_weight=grouped_weight,
        )
        self.propensity_ = propensity_value
        self.n_bins_ = actual_bins
        return self

    def predict(self, score) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("UpliftIsotonicCalibrator has not been fitted.")
        score_arr = np.asarray(score, dtype=float)
        if score_arr.ndim != 1 or not np.isfinite(score_arr).all():
            raise ValueError("score must be a one-dimensional array of finite values.")
        return np.asarray(self.model.predict(score_arr), dtype=float)


def _validate_inputs(score, y, treatment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_arr = np.asarray(score, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    treatment_arr = np.asarray(treatment, dtype=int)
    if not (
        score_arr.ndim == y_arr.ndim == treatment_arr.ndim == 1
        and len(score_arr) == len(y_arr) == len(treatment_arr)
    ):
        raise ValueError("score, y, and treatment must be one-dimensional arrays of equal length.")
    if len(score_arr) == 0:
        raise ValueError("Cannot fit the calibrator on empty data.")
    if not np.isfinite(score_arr).all():
        raise ValueError("score must contain only finite values.")
    if not set(np.unique(y_arr)).issubset({0.0, 1.0}):
        raise ValueError("y may contain only 0 and 1.")
    if not set(np.unique(treatment_arr)).issubset({0, 1}):
        raise ValueError("treatment may contain only 0 and 1.")
    return score_arr, y_arr, treatment_arr
