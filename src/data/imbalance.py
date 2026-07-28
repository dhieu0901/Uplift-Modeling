from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UndersampledUpliftData:
    """Treatment-stratified sample with all positives retained."""

    X: pd.DataFrame
    y: pd.Series
    treatment: pd.Series
    factor: float
    original_size: int
    sampled_size: int
    negative_keep_rates: tuple[float, float]


def stratified_negative_undersample(
    X: pd.DataFrame,
    y: pd.Series,
    treatment: pd.Series,
    factor: float,
    random_state: int = 42,
) -> UndersampledUpliftData:
    """Undersample negative outcomes separately within each treatment arm.

    This follows Nyberg, Kusmierczyk, and Klami (2021). The same overall
    reduction factor is applied to both arms, while the negative keep rate is
    allowed to differ so that each arm's positive rate is multiplied by
    approximately ``factor``. All positive observations are retained.
    """
    if not (len(X) == len(y) == len(treatment)):
        raise ValueError("X, y, and treatment must have equal length.")
    if len(X) == 0:
        raise ValueError("Cannot undersample an empty dataset.")
    factor_value = float(factor)
    if not np.isfinite(factor_value) or factor_value < 1.0:
        raise ValueError("factor must be a finite number greater than or equal to 1.")

    y_arr = y.to_numpy(dtype=int)
    treatment_arr = treatment.to_numpy(dtype=int)
    if not set(np.unique(y_arr)).issubset({0, 1}):
        raise ValueError("y may contain only 0 and 1.")
    if set(np.unique(treatment_arr)) != {0, 1}:
        raise ValueError("Both treatment arms are required.")

    rng = np.random.default_rng(random_state)
    selected_parts = []
    negative_keep_rates = []
    for treatment_value in (0, 1):
        arm_indices = np.flatnonzero(treatment_arr == treatment_value)
        positive_indices = arm_indices[y_arr[arm_indices] == 1]
        negative_indices = arm_indices[y_arr[arm_indices] == 0]
        if positive_indices.size == 0 or negative_indices.size == 0:
            raise ValueError(
                "Every treatment arm must contain positive and negative outcomes."
            )

        positive_rate = positive_indices.size / arm_indices.size
        maximum_factor = 1.0 / positive_rate
        if factor_value >= maximum_factor and factor_value != 1.0:
            raise ValueError(
                f"factor must be smaller than {maximum_factor:.6f} for "
                f"treatment arm {treatment_value}."
            )
        negative_keep_fraction = (
            1.0 / factor_value - positive_rate
        ) / (1.0 - positive_rate)
        negative_keep_fraction = float(
            np.clip(negative_keep_fraction, 0.0, 1.0)
        )
        n_negative_keep = int(
            round(negative_keep_fraction * negative_indices.size)
        )
        if factor_value == 1.0:
            kept_negative = negative_indices
        else:
            n_negative_keep = max(1, n_negative_keep)
            kept_negative = rng.choice(
                negative_indices,
                size=n_negative_keep,
                replace=False,
            )
        negative_keep_rates.append(
            float(len(kept_negative) / len(negative_indices))
        )
        selected_parts.extend([positive_indices, kept_negative])

    selected = np.concatenate(selected_parts)
    selected = rng.permutation(selected)
    return UndersampledUpliftData(
        X=X.iloc[selected].reset_index(drop=True),
        y=y.iloc[selected].reset_index(drop=True),
        treatment=treatment.iloc[selected].reset_index(drop=True),
        factor=factor_value,
        original_size=len(X),
        sampled_size=len(selected),
        negative_keep_rates=(
            negative_keep_rates[0],
            negative_keep_rates[1],
        ),
    )
