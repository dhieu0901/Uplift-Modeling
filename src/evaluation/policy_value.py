from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import NormalDist

import numpy as np
import numpy.typing as npt
import pandas as pd


def uplift_score_threshold(
    outcome_value: float,
    treatment_cost: float,
) -> float:
    """Break-even uplift threshold based on outcome value and treatment cost.

    A user has positive expected net value when
    ``uplift_score * outcome_value > treatment_cost``.
    """
    outcome_value, treatment_cost = _validate_unit_economics(
        outcome_value, treatment_cost
    )
    return treatment_cost / outcome_value


def target_by_expected_value(
    uplift_score: Sequence[float],
    outcome_value: float,
    treatment_cost: float,
) -> np.ndarray:
    """Return a binary policy based on user-level expected net value."""
    score = np.asarray(uplift_score, dtype=float)
    if not np.isfinite(score).all():
        raise ValueError("uplift_score must contain only finite values.")
    threshold = uplift_score_threshold(outcome_value, treatment_cost)
    return score > threshold


def doubly_robust_treatment_effect_scores(
    y: Sequence[float],
    treatment: Sequence[int],
    mu0: Sequence[float],
    mu1: Sequence[float],
    propensity: float,
) -> np.ndarray:
    """Build AIPW pseudo-outcomes whose conditional mean is the CATE.

    The augmented inverse-probability-weighted form is due to Robins,
    Rotnitzky, and Zhao (1994), *JASA* 89(427):846-866. It is doubly robust:
    unbiased if either the outcome models or the propensity is correct, and in
    a randomized design the propensity is known, so the property holds by
    construction.

    ``mu0`` and ``mu1`` must be predictions from models that did not train on
    the evaluation outcomes. In an RCT, ``propensity`` is the known or locked
    treatment-assignment probability.
    """
    y_arr, treatment_arr, mu0_arr, mu1_arr, propensity_value = (
        _validate_aipw_inputs(y, treatment, mu0, mu1, propensity)
    )
    return (
        mu1_arr
        - mu0_arr
        + treatment_arr
        / propensity_value
        * (y_arr - mu1_arr)
        - (1.0 - treatment_arr)
        / (1.0 - propensity_value)
        * (y_arr - mu0_arr)
    )


def aipw_policy_value_table(
    y: Sequence[float],
    treatment: Sequence[int],
    scores: Mapping[str, npt.ArrayLike],
    mu0: Sequence[float],
    mu1: Sequence[float],
    propensity: float,
    fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
    confidence_level: float = 0.95,
    outcome_value: float | None = None,
    treatment_cost: float = 0.0,
) -> pd.DataFrame:
    """Estimate incremental outcomes for fixed-budget policies using AIPW.

    The estimand is the value of each targeting policy relative to treating
    nobody. This is the binary-action specialization of doubly robust policy
    value. Confidence intervals use the influence-score standard error and
    condition on the already-fitted policy.
    """
    pseudo_outcome = doubly_robust_treatment_effect_scores(
        y, treatment, mu0, mu1, propensity
    )
    n = len(pseudo_outcome)
    z_value = _normal_critical_value(confidence_level)
    score_arrays = _validate_policy_scores(scores, n)
    _validate_fractions(fractions)
    if outcome_value is not None:
        outcome_value, treatment_cost = _validate_unit_economics(
            outcome_value, treatment_cost
        )

    rows = []
    for policy, score in score_arrays.items():
        order = np.argsort(-score, kind="mergesort")
        for fraction in fractions:
            n_targeted = max(1, int(round(float(fraction) * n)))
            target = np.zeros(n, dtype=float)
            target[order[:n_targeted]] = 1.0
            influence = target * pseudo_outcome
            estimate_rate, standard_error_rate = _mean_and_standard_error(
                influence
            )
            ci_lower_rate = estimate_rate - z_value * standard_error_rate
            ci_upper_rate = estimate_rate + z_value * standard_error_rate
            row = {
                "policy": policy,
                "budget_pct": round(100.0 * float(fraction), 4),
                "n_targeted": n_targeted,
                "incremental_outcome_rate": estimate_rate,
                "standard_error_rate": standard_error_rate,
                "ci_lower_rate": ci_lower_rate,
                "ci_upper_rate": ci_upper_rate,
                "incremental_outcome": n * estimate_rate,
                "ci_lower": n * ci_lower_rate,
                "ci_upper": n * ci_upper_rate,
            }
            if outcome_value is not None:
                campaign_cost = n_targeted * treatment_cost
                row.update(
                    {
                        "gross_value": n * estimate_rate * outcome_value,
                        "campaign_cost": campaign_cost,
                        "net_value": n
                        * estimate_rate
                        * outcome_value
                        - campaign_cost,
                        "net_value_ci_lower": n
                        * ci_lower_rate
                        * outcome_value
                        - campaign_cost,
                        "net_value_ci_upper": n
                        * ci_upper_rate
                        * outcome_value
                        - campaign_cost,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def aipw_policy_contrast_table(
    y: Sequence[float],
    treatment: Sequence[int],
    scores: Mapping[str, npt.ArrayLike],
    reference_policy: str,
    mu0: Sequence[float],
    mu1: Sequence[float],
    propensity: float,
    fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Return paired AIPW contrasts against a reference ranking policy."""
    pseudo_outcome = doubly_robust_treatment_effect_scores(
        y, treatment, mu0, mu1, propensity
    )
    n = len(pseudo_outcome)
    score_arrays = _validate_policy_scores(scores, n)
    if reference_policy not in score_arrays:
        raise ValueError(f"Unknown reference policy: {reference_policy}")
    _validate_fractions(fractions)
    z_value = _normal_critical_value(confidence_level)
    orders = {
        policy: np.argsort(-score, kind="mergesort")
        for policy, score in score_arrays.items()
    }

    rows = []
    for fraction in fractions:
        n_targeted = max(1, int(round(float(fraction) * n)))
        reference_target = np.zeros(n, dtype=float)
        reference_target[orders[reference_policy][:n_targeted]] = 1.0
        for policy in score_arrays:
            if policy == reference_policy:
                continue
            target = np.zeros(n, dtype=float)
            target[orders[policy][:n_targeted]] = 1.0
            influence_difference = (
                target - reference_target
            ) * pseudo_outcome
            difference_rate, standard_error_rate = _mean_and_standard_error(
                influence_difference
            )
            rows.append(
                {
                    "policy": policy,
                    "reference_policy": reference_policy,
                    "budget_pct": round(100.0 * float(fraction), 4),
                    "n_targeted": n_targeted,
                    "difference_rate": difference_rate,
                    "standard_error_rate": standard_error_rate,
                    "difference": n * difference_rate,
                    "ci_lower": n
                    * (difference_rate - z_value * standard_error_rate),
                    "ci_upper": n
                    * (difference_rate + z_value * standard_error_rate),
                }
            )
    return pd.DataFrame(rows)


def monetize_policy_table(
    policy_table: pd.DataFrame,
    outcome_value: float,
    treatment_cost: float,
) -> pd.DataFrame:
    """Convert estimated incremental outcomes into gross and net value."""
    outcome_value, treatment_cost = _validate_unit_economics(
        outcome_value, treatment_cost
    )
    required_columns = {
        "policy",
        "budget_pct",
        "n_targeted",
        "incremental_outcome",
    }
    missing = sorted(required_columns - set(policy_table.columns))
    if missing:
        raise ValueError(f"Policy table is missing required columns: {missing}")

    result = policy_table.copy()
    n_targeted = pd.to_numeric(result["n_targeted"], errors="raise").to_numpy(
        dtype=float
    )
    incremental = pd.to_numeric(
        result["incremental_outcome"], errors="raise"
    ).to_numpy(dtype=float)
    if (n_targeted < 0).any():
        raise ValueError("n_targeted must not be negative.")

    gross_value = incremental * outcome_value
    campaign_cost = n_targeted * treatment_cost
    net_value = gross_value - campaign_cost
    result["outcome_value"] = outcome_value
    result["treatment_cost_per_target"] = treatment_cost
    result["gross_value"] = gross_value
    result["campaign_cost"] = campaign_cost
    result["net_value"] = net_value
    result["net_value_per_1k_targeted"] = np.divide(
        net_value * 1000,
        n_targeted,
        out=np.full_like(net_value, np.nan),
        where=n_targeted > 0,
    )
    result["break_even_cost_per_target"] = np.divide(
        gross_value,
        n_targeted,
        out=np.full_like(gross_value, np.nan),
        where=n_targeted > 0,
    )
    result["minimum_value_to_cost_ratio"] = np.divide(
        n_targeted,
        incremental,
        out=np.full_like(incremental, np.inf),
        where=incremental > 0,
    )
    result["profitable"] = net_value > 0
    return result


def _validate_unit_economics(
    outcome_value: float,
    treatment_cost: float,
) -> tuple[float, float]:
    outcome_value = float(outcome_value)
    treatment_cost = float(treatment_cost)
    if not np.isfinite(outcome_value) or outcome_value <= 0:
        raise ValueError("outcome_value must be a finite number greater than zero.")
    if not np.isfinite(treatment_cost) or treatment_cost < 0:
        raise ValueError("treatment_cost must be a finite nonnegative number.")
    return outcome_value, treatment_cost


def _validate_aipw_inputs(
    y: Sequence[float],
    treatment: Sequence[int],
    mu0: Sequence[float],
    mu1: Sequence[float],
    propensity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    arrays = [
        np.asarray(values, dtype=float)
        for values in (y, treatment, mu0, mu1)
    ]
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("y, treatment, mu0, and mu1 must be one-dimensional.")
    if len({len(array) for array in arrays}) != 1 or len(arrays[0]) == 0:
        raise ValueError(
            "y, treatment, mu0, and mu1 must have the same nonzero length."
        )
    y_arr, treatment_arr, mu0_arr, mu1_arr = arrays
    if not set(np.unique(y_arr)).issubset({0.0, 1.0}):
        raise ValueError("y may contain only 0 and 1.")
    if not set(np.unique(treatment_arr)).issubset({0.0, 1.0}):
        raise ValueError("treatment may contain only 0 and 1.")
    if not np.isfinite(mu0_arr).all() or not np.isfinite(mu1_arr).all():
        raise ValueError("mu0 and mu1 must contain only finite values.")
    propensity_value = float(propensity)
    if not np.isfinite(propensity_value) or not 0.0 < propensity_value < 1.0:
        raise ValueError("propensity must be in the interval (0, 1).")
    return y_arr, treatment_arr, mu0_arr, mu1_arr, propensity_value


def _validate_policy_scores(
    scores: Mapping[str, npt.ArrayLike],
    expected_length: int,
) -> dict[str, np.ndarray]:
    if not scores:
        raise ValueError("scores must contain at least one policy.")
    result = {}
    for policy, score in scores.items():
        score_arr = np.asarray(score, dtype=float)
        if score_arr.ndim != 1 or len(score_arr) != expected_length:
            raise ValueError(
                f"Score for policy {policy} must be one-dimensional and "
                "match the evaluation sample."
            )
        if not np.isfinite(score_arr).all():
            raise ValueError(f"Score for policy {policy} must be finite.")
        result[policy] = score_arr
    return result


def _validate_fractions(fractions: Sequence[float]) -> None:
    if not fractions:
        raise ValueError("fractions must not be empty.")
    if any(not 0.0 < float(fraction) <= 1.0 for fraction in fractions):
        raise ValueError("Every policy fraction must be in the interval (0, 1].")


def _normal_critical_value(confidence_level: float) -> float:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in the interval (0, 1).")
    return NormalDist().inv_cdf(0.5 + confidence_level / 2.0)


def _mean_and_standard_error(values: np.ndarray) -> tuple[float, float]:
    estimate = float(values.mean())
    if len(values) < 2:
        return estimate, 0.0
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    return estimate, standard_error
