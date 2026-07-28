from __future__ import annotations

from collections.abc import Sequence
from statistics import NormalDist

import numpy as np
import pandas as pd


def uplift_calibration_table(
    y: Sequence[float],
    treatment: Sequence[int],
    predicted_uplift: Sequence[float],
    n_bins: int = 10,
    confidence_level: float = 0.95,
    binning_score: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Compare predicted and observed uplift across score quantiles.

    ``binning_score`` lets raw and calibrated scores use exactly the same user
    groups, which is especially useful when isotonic calibration creates many ties.
    """
    y_arr = np.asarray(y, dtype=float)
    treatment_arr = np.asarray(treatment, dtype=int)
    score_arr = np.asarray(predicted_uplift, dtype=float)
    _validate_inputs(y_arr, treatment_arr, score_arr, n_bins, confidence_level)
    ranking_arr = (
        score_arr
        if binning_score is None
        else np.asarray(binning_score, dtype=float)
    )
    if ranking_arr.shape != score_arr.shape or not np.isfinite(ranking_arr).all():
        raise ValueError("binning_score must match predicted_uplift in shape and contain finite values.")

    order = np.argsort(ranking_arr, kind="mergesort")[::-1]
    y_sorted = y_arr[order]
    treatment_sorted = treatment_arr[order]
    score_sorted = score_arr[order]
    actual_bins = min(n_bins, len(y_arr))
    bin_ids = np.floor(
        np.arange(len(y_arr), dtype=float) * actual_bins / len(y_arr)
    ).astype(int)
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)

    rows = []
    for bin_id in range(actual_bins):
        mask = bin_ids == bin_id
        y_bin = y_sorted[mask]
        treatment_bin = treatment_sorted[mask]
        score_bin = score_sorted[mask]
        treated = treatment_bin == 1
        control = treatment_bin == 0
        n_treated = int(treated.sum())
        n_control = int(control.sum())

        if n_treated == 0 or n_control == 0:
            treated_rate = control_rate = observed_uplift = standard_error = float(
                "nan"
            )
        else:
            treated_rate = float(y_bin[treated].mean())
            control_rate = float(y_bin[control].mean())
            observed_uplift = treated_rate - control_rate
            standard_error = float(
                np.sqrt(
                    treated_rate * (1.0 - treated_rate) / n_treated
                    + control_rate * (1.0 - control_rate) / n_control
                )
            )

        predicted_mean = float(score_bin.mean())
        rows.append(
            {
                "bin": bin_id + 1,
                "n": int(mask.sum()),
                "treated_n": n_treated,
                "control_n": n_control,
                "score_min": float(score_bin.min()),
                "score_max": float(score_bin.max()),
                "predicted_uplift": predicted_mean,
                "treated_outcome_rate": treated_rate,
                "control_outcome_rate": control_rate,
                "observed_uplift": observed_uplift,
                "observed_standard_error": standard_error,
                "ci_lower": observed_uplift - z_value * standard_error,
                "ci_upper": observed_uplift + z_value * standard_error,
                "calibration_error": observed_uplift - predicted_mean,
            }
        )
    return pd.DataFrame(rows)


def summarize_uplift_calibration(calibration_table: pd.DataFrame) -> dict[str, float]:
    """Summarize calibration error and the regression of observed on predicted uplift."""
    required = {"n", "predicted_uplift", "observed_uplift"}
    missing = sorted(required - set(calibration_table.columns))
    if missing:
        raise ValueError(f"Calibration table is missing columns: {missing}")

    clean = calibration_table.dropna(
        subset=["n", "predicted_uplift", "observed_uplift"]
    )
    if clean.empty:
        return {
            "weighted_bias": float("nan"),
            "weighted_mae": float("nan"),
            "weighted_rmse": float("nan"),
            "euce": float("nan"),
            "muce": float("nan"),
            "calibration_intercept": float("nan"),
            "calibration_slope": float("nan"),
        }

    weights = clean["n"].to_numpy(dtype=float)
    predicted = clean["predicted_uplift"].to_numpy(dtype=float)
    observed = clean["observed_uplift"].to_numpy(dtype=float)
    errors = observed - predicted
    weighted_bias = float(np.average(errors, weights=weights))
    weighted_mae = float(np.average(np.abs(errors), weights=weights))
    weighted_rmse = float(np.sqrt(np.average(errors**2, weights=weights)))
    muce = float(np.max(np.abs(errors)))

    if np.ptp(predicted) == 0:
        intercept = slope = float("nan")
    else:
        design = np.column_stack([np.ones_like(predicted), predicted])
        sqrt_weights = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(
            design * sqrt_weights[:, None],
            observed * sqrt_weights,
            rcond=None,
        )
        intercept, slope = (float(value) for value in coefficients)

    return {
        "weighted_bias": weighted_bias,
        "weighted_mae": weighted_mae,
        "weighted_rmse": weighted_rmse,
        "euce": weighted_mae,
        "muce": muce,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _validate_inputs(
    y: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    n_bins: int,
    confidence_level: float,
) -> None:
    if not (y.ndim == treatment.ndim == score.ndim == 1 and len(y) == len(treatment) == len(score)):
        raise ValueError("y, treatment, and predicted_uplift must be one-dimensional arrays of equal length.")
    if len(y) == 0:
        raise ValueError("Cannot evaluate calibration on empty data.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in the interval (0, 1).")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("y may contain only 0 and 1.")
    if not set(np.unique(treatment)).issubset({0, 1}):
        raise ValueError("treatment may contain only 0 and 1.")
    if not np.isfinite(score).all():
        raise ValueError("predicted_uplift must contain only finite values.")
