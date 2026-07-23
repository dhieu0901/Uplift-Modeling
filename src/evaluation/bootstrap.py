from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.uplift import budget_policy_table, separate_relative_auuc


@dataclass(frozen=True)
class BootstrapUpliftResult:
    """Confidence intervals for ranking metrics and business gains."""

    policy_metrics: pd.DataFrame
    business_gains: pd.DataFrame


def bootstrap_uplift_uncertainty(
    y: Sequence[float],
    treatment: Sequence[int],
    scores: Mapping[str, Sequence[float]],
    champion: str,
    response_policy: str = "response_model",
    random_policy: str = "random",
    fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
    n_bootstraps: int = 200,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> BootstrapUpliftResult:
    """Bootstrap by treatment arm and calculate percentile confidence intervals.

    Resampling treated and control observations separately preserves the two arm
    sizes in every bootstrap sample. Policy differences are calculated on the
    same bootstrap sample, so the confidence intervals are paired.
    """
    if n_bootstraps < 2:
        raise ValueError("n_bootstraps must be at least 2.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in the interval (0, 1).")

    y_arr = np.asarray(y, dtype=int)
    w_arr = np.asarray(treatment, dtype=int)
    score_arrays = {name: np.asarray(score, dtype=float) for name, score in scores.items()}
    _validate_inputs(y_arr, w_arr, score_arrays)

    required_policies = {champion, response_policy, random_policy}
    missing = sorted(required_policies - set(score_arrays))
    if missing:
        raise ValueError(f"Missing scores for policies: {missing}")

    point_metrics = {
        policy: separate_relative_auuc(y_arr, w_arr, score)
        for policy, score in score_arrays.items()
    }
    budget_scores = {
        policy: score_arrays[policy]
        for policy in [champion, response_policy, random_policy]
    }
    point_budget = budget_policy_table(y_arr, w_arr, budget_scores, fractions=fractions)
    point_pivot = point_budget.pivot(
        index="budget_pct", columns="policy", values="incremental_outcome"
    )

    metric_samples: dict[str, list[float]] = {policy: [] for policy in score_arrays}
    business_samples = {
        round(100 * fraction, 1): {
            "champion": [],
            "response": [],
            "random": [],
            "gain_vs_response": [],
            "gain_vs_random": [],
        }
        for fraction in fractions
    }
    rng = np.random.default_rng(random_state)

    for _ in range(n_bootstraps):
        sample_indices = _resample_within_treatment(w_arr, rng)
        y_sample = y_arr[sample_indices]
        w_sample = w_arr[sample_indices]
        sampled_scores = {
            policy: score[sample_indices] for policy, score in score_arrays.items()
        }

        for policy, score in sampled_scores.items():
            metric_samples[policy].append(
                separate_relative_auuc(y_sample, w_sample, score)
            )

        sampled_budget = budget_policy_table(
            y_sample,
            w_sample,
            {
                champion: sampled_scores[champion],
                response_policy: sampled_scores[response_policy],
                random_policy: sampled_scores[random_policy],
            },
            fractions=fractions,
        ).pivot(index="budget_pct", columns="policy", values="incremental_outcome")

        for budget_pct, values in business_samples.items():
            champion_value = float(sampled_budget.loc[budget_pct, champion])
            response_value = float(sampled_budget.loc[budget_pct, response_policy])
            random_value = float(sampled_budget.loc[budget_pct, random_policy])
            values["champion"].append(champion_value)
            values["response"].append(response_value)
            values["random"].append(random_value)
            values["gain_vs_response"].append(champion_value - response_value)
            values["gain_vs_random"].append(champion_value - random_value)

    policy_rows = []
    for policy, samples in metric_samples.items():
        summary = _summarize_samples(samples, confidence_level)
        difference_samples = np.asarray(samples) - np.asarray(metric_samples[response_policy])
        difference_summary = _summarize_samples(difference_samples, confidence_level)
        policy_rows.append(
            {
                "policy": policy,
                "estimate": point_metrics[policy],
                "bootstrap_mean": summary["mean"],
                "bootstrap_std": summary["std"],
                "ci_lower": summary["lower"],
                "ci_upper": summary["upper"],
                "difference_vs_response": point_metrics[policy]
                - point_metrics[response_policy],
                "difference_ci_lower": difference_summary["lower"],
                "difference_ci_upper": difference_summary["upper"],
                "n_valid": summary["n_valid"],
            }
        )

    business_rows = []
    for budget_pct, samples in business_samples.items():
        champion_ci = _summarize_samples(samples["champion"], confidence_level)
        response_gain_ci = _summarize_samples(samples["gain_vs_response"], confidence_level)
        random_gain_ci = _summarize_samples(samples["gain_vs_random"], confidence_level)
        champion_point = float(point_pivot.loc[budget_pct, champion])
        response_point = float(point_pivot.loc[budget_pct, response_policy])
        random_point = float(point_pivot.loc[budget_pct, random_policy])
        business_rows.append(
            {
                "budget_pct": budget_pct,
                "champion_estimate": champion_point,
                "champion_ci_lower": champion_ci["lower"],
                "champion_ci_upper": champion_ci["upper"],
                "gain_vs_response": champion_point - response_point,
                "gain_vs_response_ci_lower": response_gain_ci["lower"],
                "gain_vs_response_ci_upper": response_gain_ci["upper"],
                "gain_vs_random": champion_point - random_point,
                "gain_vs_random_ci_lower": random_gain_ci["lower"],
                "gain_vs_random_ci_upper": random_gain_ci["upper"],
            }
        )

    return BootstrapUpliftResult(
        policy_metrics=pd.DataFrame(policy_rows).sort_values("estimate", ascending=False),
        business_gains=pd.DataFrame(business_rows),
    )


def _resample_within_treatment(
    treatment: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    arms = []
    for treatment_value in [0, 1]:
        indices = np.flatnonzero(treatment == treatment_value)
        if indices.size == 0:
            raise ValueError("Bootstrap requires both treatment and control observations.")
        arms.append(rng.choice(indices, size=indices.size, replace=True))
    return np.concatenate(arms)


def _summarize_samples(samples: Sequence[float], confidence_level: float) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "n_valid": 0,
        }

    alpha = (1.0 - confidence_level) / 2.0
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
        "n_valid": int(values.size),
    }


def _validate_inputs(
    y: np.ndarray,
    treatment: np.ndarray,
    scores: Mapping[str, np.ndarray],
) -> None:
    if len(y) != len(treatment):
        raise ValueError("y and treatment must have equal length.")
    if not set(np.unique(treatment)).issubset({0, 1}):
        raise ValueError("treatment may contain only 0 and 1.")
    for policy, score in scores.items():
        if len(score) != len(y):
            raise ValueError(f"Score for policy {policy} has an incorrect length.")
