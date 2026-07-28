from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ground_truth_cate_metrics(
    true_cate: Sequence[float],
    scores: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    """Evaluate CATE estimates when semi-synthetic ground truth is known."""
    truth = np.asarray(true_cate, dtype=float)
    score_arrays = _validate_scores(truth, scores)
    rows = []
    for policy, score in score_arrays.items():
        error = score - truth
        rows.append(
            {
                "policy": policy,
                "pehe": float(np.sqrt(np.mean(error**2))),
                "cate_mae": float(np.mean(np.abs(error))),
                "cate_bias": float(np.mean(error)),
                "pearson": _safe_correlation(truth, score),
                "spearman": float(spearmanr(truth, score).statistic),
            }
        )
    return pd.DataFrame(rows).sort_values("pehe")


def ground_truth_policy_table(
    true_cate: Sequence[float],
    scores: Mapping[str, Sequence[float]],
    fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """Calculate exact policy gain and regret from known CATEs."""
    truth = np.asarray(true_cate, dtype=float)
    score_arrays = _validate_scores(truth, scores)
    oracle_order = np.argsort(-truth, kind="mergesort")
    n = len(truth)
    rows = []
    for fraction in fractions:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("fractions must be in the interval (0, 1].")
        n_targeted = max(1, int(round(float(fraction) * n)))
        oracle_gain = float(truth[oracle_order[:n_targeted]].sum())
        for policy, score in score_arrays.items():
            selected = np.argsort(-score, kind="mergesort")[:n_targeted]
            policy_gain = float(truth[selected].sum())
            regret = oracle_gain - policy_gain
            rows.append(
                {
                    "policy": policy,
                    "budget_pct": round(100.0 * float(fraction), 4),
                    "n_targeted": n_targeted,
                    "true_incremental_outcome": policy_gain,
                    "oracle_incremental_outcome": oracle_gain,
                    "policy_regret": regret,
                    "oracle_value_fraction": (
                        policy_gain / oracle_gain
                        if oracle_gain != 0.0
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _validate_scores(
    truth: np.ndarray,
    scores: Mapping[str, Sequence[float]],
) -> dict[str, np.ndarray]:
    if truth.ndim != 1 or len(truth) == 0 or not np.isfinite(truth).all():
        raise ValueError("true_cate must be a nonempty finite vector.")
    if not scores:
        raise ValueError("scores must contain at least one policy.")
    result = {}
    for policy, score in scores.items():
        values = np.asarray(score, dtype=float)
        if values.shape != truth.shape or not np.isfinite(values).all():
            raise ValueError(
                f"Score for policy {policy} must match true_cate and be finite."
            )
        result[policy] = values
    return result


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])
