from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def incremental_outcome(
    y: Sequence[float],
    treatment: Sequence[int],
) -> float:
    """Estimate incremental outcomes in a selected group.

    Formula:
        (mean outcome treated - mean outcome control) * selected population size

    This is the cumulative gain style estimator used in uplift evaluation.
    """
    y_arr = np.asarray(y, dtype=float)
    w_arr = np.asarray(treatment, dtype=int)
    treated_mask = w_arr == 1
    control_mask = w_arr == 0

    n_treated = int(treated_mask.sum())
    n_control = int(control_mask.sum())
    if n_treated == 0 or n_control == 0:
        return float("nan")

    uplift_rate = y_arr[treated_mask].mean() - y_arr[control_mask].mean()
    return float(uplift_rate * len(y_arr))


def cumulative_uplift_curve(
    y: Sequence[float],
    treatment: Sequence[int],
    score: Sequence[float],
    min_fraction: float = 0.01,
    n_points: int = 100,
) -> pd.DataFrame:
    """Build a cumulative uplift curve by sorting users by descending score."""
    y_arr = np.asarray(y, dtype=float)
    w_arr = np.asarray(treatment, dtype=int)
    s_arr = np.asarray(score, dtype=float)
    if not (len(y_arr) == len(w_arr) == len(s_arr)):
        raise ValueError("y, treatment, and score must have equal length.")

    order = np.argsort(-s_arr)
    y_sorted = y_arr[order]
    w_sorted = w_arr[order]

    fractions = np.linspace(min_fraction, 1.0, n_points)
    rows = [
        {
            "fraction": 0.0,
            "n_targeted": 0,
            "incremental_outcome": 0.0,
        }
    ]
    n = len(y_sorted)
    for fraction in fractions:
        k = max(1, int(round(fraction * n)))
        rows.append(
            {
                "fraction": fraction,
                "n_targeted": k,
                "incremental_outcome": incremental_outcome(y_sorted[:k], w_sorted[:k]),
            }
        )
    return pd.DataFrame(rows)


def exact_uplift_curve(
    y: Sequence[float],
    treatment: Sequence[int],
    score: Sequence[float],
) -> pd.DataFrame:
    """Calculate cumulative uplift at every distinct score threshold.

    Sorting, tied-score handling, and the curve formula match
    ``sklift.metrics.uplift_curve``. This function is suitable for library
    validation; ``cumulative_uplift_curve`` remains useful when a compact
    percentage grid is needed for plotting.
    """
    y_arr, w_arr, s_arr = _validate_binary_curve_inputs(y, treatment, score)
    threshold_indices, y_sorted, w_sorted = _exact_curve_inputs(
        y_arr, w_arr, s_arr
    )

    n_targeted = threshold_indices + 1
    n_treated = np.cumsum(w_sorted, dtype=float)[threshold_indices]
    n_control = n_targeted - n_treated
    treated_outcomes = np.cumsum(y_sorted * w_sorted, dtype=float)[
        threshold_indices
    ]
    control_outcomes = np.cumsum(y_sorted * (1 - w_sorted), dtype=float)[
        threshold_indices
    ]

    treated_rate = np.divide(
        treated_outcomes,
        n_treated,
        out=np.zeros_like(treated_outcomes),
        where=n_treated != 0,
    )
    control_rate = np.divide(
        control_outcomes,
        n_control,
        out=np.zeros_like(control_outcomes),
        where=n_control != 0,
    )
    curve_values = (treated_rate - control_rate) * n_targeted

    n_targeted = np.r_[0, n_targeted]
    curve_values = np.r_[0.0, curve_values]
    return pd.DataFrame(
        {
            "n_targeted": n_targeted.astype(int),
            "fraction": n_targeted / len(y_arr),
            "incremental_outcome": curve_values,
        }
    )


def exact_qini_curve(
    y: Sequence[float],
    treatment: Sequence[int],
    score: Sequence[float],
) -> pd.DataFrame:
    """Calculate the Qini curve at every score threshold using the scikit-uplift definition."""
    y_arr, w_arr, s_arr = _validate_binary_curve_inputs(y, treatment, score)
    threshold_indices, y_sorted, w_sorted = _exact_curve_inputs(
        y_arr, w_arr, s_arr
    )

    n_targeted = threshold_indices + 1
    n_treated = np.cumsum(w_sorted, dtype=float)[threshold_indices]
    n_control = n_targeted - n_treated
    treated_outcomes = np.cumsum(y_sorted * w_sorted, dtype=float)[
        threshold_indices
    ]
    control_outcomes = np.cumsum(y_sorted * (1 - w_sorted), dtype=float)[
        threshold_indices
    ]
    treatment_control_ratio = np.divide(
        n_treated,
        n_control,
        out=np.zeros_like(n_treated),
        where=n_control != 0,
    )
    curve_values = treated_outcomes - control_outcomes * treatment_control_ratio

    n_targeted = np.r_[0, n_targeted]
    curve_values = np.r_[0.0, curve_values]
    return pd.DataFrame(
        {
            "n_targeted": n_targeted.astype(int),
            "fraction": n_targeted / len(y_arr),
            "qini_value": curve_values,
        }
    )


def auuc(curve: pd.DataFrame) -> float:
    """Area under a cumulative uplift curve."""
    clean = curve.dropna(subset=["incremental_outcome"])
    if clean.empty:
        return float("nan")
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(clean["incremental_outcome"], clean["fraction"]))
    return float(np.trapz(clean["incremental_outcome"], clean["fraction"]))


def qini_coefficient(curve: pd.DataFrame) -> float:
    """Calculate the area between the uplift curve and a random-targeting policy.

    The random line connects (0, 0) to the incremental outcome from targeting the
    full population. A positive value indicates ranking better than random targeting.
    """
    clean = curve.dropna(subset=["fraction", "incremental_outcome"]).sort_values("fraction")
    if clean.empty:
        return float("nan")

    random_curve = clean["fraction"].to_numpy() * float(clean["incremental_outcome"].iloc[-1])
    difference = clean["incremental_outcome"].to_numpy() - random_curve
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(difference, clean["fraction"]))
    return float(np.trapz(difference, clean["fraction"]))


def separate_relative_auuc(
    y: Sequence[float],
    treatment: Sequence[int],
    score: Sequence[float],
) -> float:
    """Calculate separate relative AUUC using the official Criteo benchmark.

    This metric corresponds to ``auuc_sep_rel_prop1`` in the
    ``large-scale-ITE-UM-benchmark`` repository. It uses separate ranking AUCs
    for treated and control observations, then combines them using each group's
    outcome rate.
    """
    y_arr = np.asarray(y, dtype=int)
    w_arr = np.asarray(treatment, dtype=int)
    s_arr = np.asarray(score, dtype=float)
    if not (len(y_arr) == len(w_arr) == len(s_arr)):
        raise ValueError("y, treatment, and score must have equal length.")
    if not set(np.unique(w_arr)).issubset({0, 1}):
        raise ValueError("treatment may contain only 0 and 1.")

    treated = w_arr == 1
    control = w_arr == 0
    if (
        treated.sum() == 0
        or control.sum() == 0
        or np.unique(y_arr[treated]).size < 2
        or np.unique(y_arr[control]).size < 2
    ):
        return float("nan")

    mean_treated = float(y_arr[treated].mean())
    mean_control = float(y_arr[control].mean())
    lambda_treated = mean_treated * (1.0 - mean_treated)
    lambda_control = mean_control * (1.0 - mean_control)
    auc_treated = roc_auc_score(y_arr[treated], s_arr[treated])
    auc_control = roc_auc_score(y_arr[control], s_arr[control])

    return float(
        lambda_treated * auc_treated
        - lambda_control * auc_control
        + 0.5 * mean_treated**2
        - 0.5 * mean_control**2
    )


def budget_policy_table(
    y: Sequence[float],
    treatment: Sequence[int],
    scores: Mapping[str, Sequence[float]],
    fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """Compare targeting policies at fixed budget fractions."""
    rows = []
    y_arr = np.asarray(y, dtype=float)
    w_arr = np.asarray(treatment, dtype=int)
    n = len(y_arr)

    for policy, score in scores.items():
        s_arr = np.asarray(score, dtype=float)
        if len(s_arr) != n:
            raise ValueError(f"Score for policy {policy} has an incorrect length.")
        order = np.argsort(-s_arr)
        for fraction in fractions:
            k = max(1, int(round(fraction * n)))
            selected = order[:k]
            inc = incremental_outcome(y_arr[selected], w_arr[selected])
            rows.append(
                {
                    "policy": policy,
                    "budget_pct": round(100 * fraction, 1),
                    "n_targeted": k,
                    "treated_n": int((w_arr[selected] == 1).sum()),
                    "control_n": int((w_arr[selected] == 0).sum()),
                    "incremental_outcome": inc,
                    "incremental_outcome_per_1k": inc / k * 1000
                    if np.isfinite(inc)
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def uplift_by_quantile(
    y: Sequence[float],
    treatment: Sequence[int],
    score: Sequence[float],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Estimate uplift rate and incremental outcome in score quantiles."""
    df = pd.DataFrame({"y": y, "treatment": treatment, "score": score})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["bin"] = pd.qcut(df.index + 1, q=n_bins, labels=False)

    rows = []
    for bin_id, group in df.groupby("bin", observed=True):
        treated = group[group["treatment"] == 1]
        control = group[group["treatment"] == 0]
        if treated.empty or control.empty:
            uplift_rate = float("nan")
            inc = float("nan")
        else:
            uplift_rate = treated["y"].mean() - control["y"].mean()
            inc = uplift_rate * len(group)
        rows.append(
            {
                "bin": int(bin_id) + 1,
                "n": len(group),
                "treated_n": len(treated),
                "control_n": len(control),
                "treated_outcome_rate": treated["y"].mean() if not treated.empty else float("nan"),
                "control_outcome_rate": control["y"].mean() if not control.empty else float("nan"),
                "uplift_rate": uplift_rate,
                "incremental_outcome": inc,
            }
        )
    return pd.DataFrame(rows)


def _validate_binary_curve_inputs(
    y: Sequence[float],
    treatment: Sequence[int],
    score: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=float)
    w_arr = np.asarray(treatment, dtype=int)
    s_arr = np.asarray(score, dtype=float)
    if not (len(y_arr) == len(w_arr) == len(s_arr)):
        raise ValueError("y, treatment, and score must have equal length.")
    if len(y_arr) == 0:
        raise ValueError("Cannot calculate a curve on empty data.")
    if not set(np.unique(y_arr)).issubset({0.0, 1.0}):
        raise ValueError("y may contain only 0 and 1.")
    if not set(np.unique(w_arr)).issubset({0, 1}):
        raise ValueError("treatment may contain only 0 and 1.")
    if not np.isfinite(s_arr).all():
        raise ValueError("score must contain only finite values.")
    return y_arr, w_arr, s_arr


def _exact_curve_inputs(
    y: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(score, kind="mergesort")[::-1]
    y_sorted = y[order]
    treatment_sorted = treatment[order]
    score_sorted = score[order]
    distinct_value_indices = np.where(np.diff(score_sorted))[0]
    threshold_indices = np.r_[distinct_value_indices, len(score_sorted) - 1]
    return threshold_indices, y_sorted, treatment_sorted
