from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.models.base import predict_positive_proba


def cross_fitted_potential_outcomes(
    X: pd.DataFrame,
    y: pd.Series,
    treatment: pd.Series,
    model_factory: Callable[[int], object],
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return out-of-fold estimates of E[Y(0)|X] and E[Y(1)|X]."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if not (len(X) == len(y) == len(treatment)):
        raise ValueError("X, y, and treatment must have equal length.")

    y_arr = y.to_numpy(dtype=int)
    treatment_arr = treatment.to_numpy(dtype=int)
    strata = treatment.astype(str) + "_" + y.astype(str)
    smallest_stratum = int(strata.value_counts().min())
    actual_splits = min(n_splits, smallest_stratum)
    if actual_splits < 2:
        raise ValueError(
            "Cross-fitting requires at least two observations in every "
            "treatment/outcome stratum."
        )

    mu0 = np.empty(len(X), dtype=float)
    mu1 = np.empty(len(X), dtype=float)
    splitter = StratifiedKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=random_state,
    )

    for fold, (fit_indices, holdout_indices) in enumerate(
        splitter.split(X, strata)
    ):
        fit_treatment = treatment_arr[fit_indices]
        holdout_X = X.iloc[holdout_indices]
        for treatment_value, target in ((0, mu0), (1, mu1)):
            arm_indices = fit_indices[fit_treatment == treatment_value]
            arm_y = y_arr[arm_indices]
            if arm_indices.size == 0:
                raise ValueError(
                    "Every cross-fitting fold must contain both treatment arms."
                )
            if np.unique(arm_y).size == 1:
                prediction = np.full(
                    len(holdout_indices),
                    float(arm_y[0]),
                    dtype=float,
                )
            else:
                model = model_factory(random_state + 10 * fold + treatment_value)
                model.fit(X.iloc[arm_indices], arm_y)
                prediction = predict_positive_proba(model, holdout_X)
            target[holdout_indices] = np.clip(prediction, 0.0, 1.0)

    return mu0, mu1
