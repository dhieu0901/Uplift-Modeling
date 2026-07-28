from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from src.data.criteo import CriteoDataset


@dataclass(frozen=True)
class SemiSyntheticUpliftDataset:
    """Observed randomized data plus known response surfaces and CATE."""

    dataset: CriteoDataset
    mu0: np.ndarray
    mu1: np.ndarray
    true_cate: np.ndarray


def generate_semisynthetic_uplift(
    X: pd.DataFrame,
    control_rate: float = 0.05,
    average_effect: float = 0.015,
    heterogeneity_scale: float = 0.035,
    treatment_propensity: float = 0.50,
    random_state: int = 42,
) -> SemiSyntheticUpliftDataset:
    """Generate nonlinear binary potential outcomes on real covariates.

    The construction preserves the empirical feature distribution while
    providing known conditional response surfaces. It includes nonlinearities,
    interactions, positive and negative individual effects, and randomized
    treatment assignment.
    """
    if len(X) < 100:
        raise ValueError("Semi-synthetic generation requires at least 100 rows.")
    if X.shape[1] < 8:
        raise ValueError("Semi-synthetic generation requires at least 8 features.")
    if not 0.0 < control_rate < 1.0:
        raise ValueError("control_rate must be in the interval (0, 1).")
    if not 0.0 < treatment_propensity < 1.0:
        raise ValueError("treatment_propensity must be in the interval (0, 1).")
    if heterogeneity_scale < 0.0:
        raise ValueError("heterogeneity_scale must be nonnegative.")

    values = _robust_standardize(X)
    baseline_signal = (
        0.55 * values[:, 0]
        - 0.35 * values[:, 1]
        + 0.30 * np.sin(values[:, 2])
        + 0.18 * values[:, 3] * values[:, 4]
        - 0.12 * values[:, 5] ** 2
    )
    intercept = _calibrate_logit_intercept(baseline_signal, control_rate)
    mu0 = expit(intercept + baseline_signal)

    heterogeneity_signal = (
        0.70 * np.tanh(values[:, 5])
        - 0.45 * np.sin(values[:, 6])
        + 0.35 * (values[:, 7] > 0.0).astype(float)
        + 0.20 * values[:, 0] * values[:, 8 % values.shape[1]]
    )
    heterogeneity_signal -= heterogeneity_signal.mean()
    true_effect = average_effect + heterogeneity_scale * np.tanh(
        heterogeneity_signal
    )
    mu1 = np.clip(mu0 + true_effect, 1e-5, 1.0 - 1e-5)
    true_cate = mu1 - mu0

    rng = np.random.default_rng(random_state)
    treatment = rng.binomial(1, treatment_propensity, size=len(X))
    y0 = rng.binomial(1, mu0)
    y1 = rng.binomial(1, mu1)
    observed_y = np.where(treatment == 1, y1, y0)

    X_reset = X.reset_index(drop=True).astype("float32")
    raw = X_reset.copy()
    raw["treatment"] = treatment
    raw["synthetic_outcome"] = observed_y
    dataset = CriteoDataset(
        X=X_reset,
        y=pd.Series(observed_y, name="synthetic_outcome", dtype="int8"),
        treatment=pd.Series(treatment, name="treatment", dtype="int8"),
        raw=raw,
        feature_columns=list(X_reset.columns),
        outcome="synthetic_outcome",
    )
    return SemiSyntheticUpliftDataset(
        dataset=dataset,
        mu0=np.asarray(mu0, dtype=float),
        mu1=np.asarray(mu1, dtype=float),
        true_cate=np.asarray(true_cate, dtype=float),
    )


def _robust_standardize(X: pd.DataFrame) -> np.ndarray:
    values = X.to_numpy(dtype=float, copy=True)
    medians = np.nanmedian(values, axis=0)
    missing_rows, missing_columns = np.where(~np.isfinite(values))
    if missing_rows.size:
        values[missing_rows, missing_columns] = medians[missing_columns]
    lower = np.quantile(values, 0.25, axis=0)
    upper = np.quantile(values, 0.75, axis=0)
    scale = upper - lower
    standard_deviation = values.std(axis=0)
    scale = np.where(scale > 1e-8, scale, standard_deviation)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return np.clip((values - medians) / scale, -5.0, 5.0)


def _calibrate_logit_intercept(
    signal: np.ndarray,
    target_rate: float,
) -> float:
    lower, upper = -30.0, 30.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if float(expit(midpoint + signal).mean()) < target_rate:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0
