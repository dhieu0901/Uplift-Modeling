from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import chisquare


REQUIRED_RESULT_COLUMNS = {
    "arm",
    "assigned_n",
    "outcome_observed_n",
    "outcomes",
    "targeted_n",
    "treatment_received_n",
}
REQUIRED_DESIGN_COLUMNS = {"arm", "sample_size"}


@dataclass(frozen=True)
class OnlineExperimentAnalysis:
    """Main result tables for a randomized policy experiment."""

    arm_summary: pd.DataFrame
    srm_table: pd.DataFrame
    comparisons: pd.DataFrame
    policy_value: pd.DataFrame
    quality_checks: pd.DataFrame
    srm_statistic: float
    srm_p_value: float


def analyze_online_experiment(
    results: pd.DataFrame,
    design: pd.DataFrame,
    arm_a: str = "A",
    arm_b: str = "B",
    holdout_arm: str = "H",
    outcome_value: float = 100.0,
    treatment_cost: float = 5.0,
    confidence_level: float = 0.95,
    srm_alpha: float = 0.01,
    max_missing_rate_gap: float = 0.005,
) -> OnlineExperimentAnalysis:
    """Analyze aggregate ITT, SRM, and policy value by randomized arm."""
    clean_results, clean_design = _validate_and_prepare(results, design)
    required_arms = {arm_a, arm_b, holdout_arm}
    missing_arms = sorted(required_arms - set(clean_results["arm"]))
    if missing_arms:
        raise ValueError(f"Missing arms in results: {missing_arms}")
    if outcome_value <= 0 or treatment_cost < 0:
        raise ValueError("outcome_value must be positive and treatment_cost must not be negative.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in the interval (0, 1).")

    arm_summary = build_arm_summary(clean_results, confidence_level)
    srm_table, srm_statistic, srm_p_value = sample_ratio_mismatch(
        clean_results, clean_design
    )
    comparisons = pd.DataFrame(
        [
            binary_itt_comparison(
                clean_results, arm_a, arm_b, confidence_level, "primary_A_vs_B"
            ),
            binary_itt_comparison(
                clean_results, arm_a, holdout_arm, confidence_level, "A_vs_holdout"
            ),
            binary_itt_comparison(
                clean_results, arm_b, holdout_arm, confidence_level, "B_vs_holdout"
            ),
        ]
    )
    policy_value = build_policy_value_table(
        arm_summary,
        holdout_arm=holdout_arm,
        outcome_value=outcome_value,
        treatment_cost=treatment_cost,
    )
    comparisons = add_net_value_comparison(
        comparisons,
        policy_value,
        outcome_value=outcome_value,
    )
    quality_checks = build_quality_checks(
        arm_summary,
        srm_p_value=srm_p_value,
        srm_alpha=srm_alpha,
        holdout_arm=holdout_arm,
        max_missing_rate_gap=max_missing_rate_gap,
    )
    return OnlineExperimentAnalysis(
        arm_summary=arm_summary,
        srm_table=srm_table,
        comparisons=comparisons,
        policy_value=policy_value,
        quality_checks=quality_checks,
        srm_statistic=srm_statistic,
        srm_p_value=srm_p_value,
    )


def build_arm_summary(
    results: pd.DataFrame,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    summary = results.copy()
    summary["outcome_rate"] = summary["outcomes"] / summary["outcome_observed_n"]
    outcome_se = np.sqrt(
        summary["outcome_rate"]
        * (1.0 - summary["outcome_rate"])
        / summary["outcome_observed_n"]
    )
    summary["outcome_ci_lower"] = summary["outcome_rate"] - z_value * outcome_se
    summary["outcome_ci_upper"] = summary["outcome_rate"] + z_value * outcome_se
    summary["missing_outcome_rate"] = (
        1.0 - summary["outcome_observed_n"] / summary["assigned_n"]
    )
    summary["target_rate"] = summary["targeted_n"] / summary["assigned_n"]
    summary["treatment_received_rate"] = (
        summary["treatment_received_n"] / summary["assigned_n"]
    )
    summary["delivery_rate"] = np.divide(
        summary["treatment_received_n"],
        summary["targeted_n"],
        out=np.full(len(summary), np.nan, dtype=float),
        where=summary["targeted_n"] > 0,
    )
    if "total_campaign_cost" not in summary:
        summary["total_campaign_cost"] = np.nan
    return summary[
        [
            "arm",
            "assigned_n",
            "outcome_observed_n",
            "outcomes",
            "outcome_rate",
            "outcome_ci_lower",
            "outcome_ci_upper",
            "missing_outcome_rate",
            "targeted_n",
            "target_rate",
            "treatment_received_n",
            "treatment_received_rate",
            "delivery_rate",
            "total_campaign_cost",
        ]
    ].sort_values("arm")


def sample_ratio_mismatch(
    results: pd.DataFrame,
    design: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    planned = design[["arm", "sample_size"]].copy()
    planned["planned_share"] = planned["sample_size"] / planned["sample_size"].sum()
    observed = results[["arm", "assigned_n"]]
    table = planned.merge(observed, on="arm", how="inner", validate="one_to_one")
    if len(table) != len(planned) or len(table) != len(observed):
        raise ValueError("Arm lists do not match between design and results.")
    table["expected_n"] = table["assigned_n"].sum() * table["planned_share"]
    table["pearson_residual"] = (
        table["assigned_n"] - table["expected_n"]
    ) / np.sqrt(table["expected_n"])
    statistic, p_value = chisquare(
        table["assigned_n"].to_numpy(dtype=float),
        table["expected_n"].to_numpy(dtype=float),
    )
    return table, float(statistic), float(p_value)


def binary_itt_comparison(
    results: pd.DataFrame,
    arm_a: str,
    arm_b: str,
    confidence_level: float = 0.95,
    comparison: str | None = None,
) -> dict[str, float | int | str]:
    left = _get_arm(results, arm_a)
    right = _get_arm(results, arm_b)
    n_a = int(left["outcome_observed_n"])
    n_b = int(right["outcome_observed_n"])
    outcomes_a = int(left["outcomes"])
    outcomes_b = int(right["outcomes"])
    rate_a = outcomes_a / n_a
    rate_b = outcomes_b / n_b
    difference = rate_a - rate_b
    standard_error = float(
        np.sqrt(
            rate_a * (1.0 - rate_a) / n_a
            + rate_b * (1.0 - rate_b) / n_b
        )
    )
    z_ci = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    pooled_rate = (outcomes_a + outcomes_b) / (n_a + n_b)
    null_standard_error = float(
        np.sqrt(pooled_rate * (1.0 - pooled_rate) * (1.0 / n_a + 1.0 / n_b))
    )
    z_statistic = difference / null_standard_error if null_standard_error else 0.0
    p_value = 2.0 * (1.0 - NormalDist().cdf(abs(z_statistic)))
    return {
        "comparison": comparison or f"{arm_a}_vs_{arm_b}",
        "arm_a": arm_a,
        "arm_b": arm_b,
        "observed_n_a": n_a,
        "observed_n_b": n_b,
        "rate_a": rate_a,
        "rate_b": rate_b,
        "absolute_difference": difference,
        "relative_lift": difference / rate_b if rate_b else float("nan"),
        "standard_error": standard_error,
        "ci_lower": difference - z_ci * standard_error,
        "ci_upper": difference + z_ci * standard_error,
        "z_statistic": z_statistic,
        "p_value": p_value,
    }


def build_policy_value_table(
    arm_summary: pd.DataFrame,
    holdout_arm: str,
    outcome_value: float,
    treatment_cost: float,
) -> pd.DataFrame:
    holdout = _get_arm(arm_summary, holdout_arm)
    holdout_rate = float(holdout["outcome_rate"])
    rows = []
    for _, arm in arm_summary.iterrows():
        assigned_n = int(arm["assigned_n"])
        incremental_outcomes = (
            float(arm["outcome_rate"]) - holdout_rate
        ) * assigned_n
        if pd.notna(arm["total_campaign_cost"]):
            campaign_cost = float(arm["total_campaign_cost"])
        else:
            campaign_cost = float(arm["targeted_n"]) * treatment_cost
        if arm["arm"] == holdout_arm:
            incremental_outcomes = 0.0
            campaign_cost = 0.0
        gross_value = incremental_outcomes * outcome_value
        net_value = gross_value - campaign_cost
        rows.append(
            {
                "arm": arm["arm"],
                "incremental_outcomes_vs_holdout": incremental_outcomes,
                "gross_value": gross_value,
                "campaign_cost": campaign_cost,
                "campaign_cost_per_assigned_user": campaign_cost / assigned_n,
                "net_value": net_value,
                "net_value_per_assigned_user": net_value / assigned_n,
            }
        )
    return pd.DataFrame(rows).sort_values("arm")


def add_net_value_comparison(
    comparisons: pd.DataFrame,
    policy_value: pd.DataFrame,
    outcome_value: float,
) -> pd.DataFrame:
    indexed_value = policy_value.set_index("arm")
    result = comparisons.copy()
    net_differences = []
    net_ci_lower = []
    net_ci_upper = []
    for row in result.itertuples():
        cost_difference = float(
            indexed_value.loc[row.arm_a, "campaign_cost_per_assigned_user"]
            - indexed_value.loc[row.arm_b, "campaign_cost_per_assigned_user"]
        )
        net_differences.append(row.absolute_difference * outcome_value - cost_difference)
        net_ci_lower.append(row.ci_lower * outcome_value - cost_difference)
        net_ci_upper.append(row.ci_upper * outcome_value - cost_difference)
    result["net_value_difference_per_user"] = net_differences
    result["net_value_difference_ci_lower"] = net_ci_lower
    result["net_value_difference_ci_upper"] = net_ci_upper
    return result


def build_quality_checks(
    arm_summary: pd.DataFrame,
    srm_p_value: float,
    srm_alpha: float,
    holdout_arm: str,
    max_missing_rate_gap: float,
) -> pd.DataFrame:
    missing_rates = arm_summary["missing_outcome_rate"]
    missing_gap = float(missing_rates.max() - missing_rates.min())
    holdout = _get_arm(arm_summary, holdout_arm)
    return pd.DataFrame(
        [
            {
                "check": "sample_ratio_mismatch",
                "value": srm_p_value,
                "threshold": srm_alpha,
                "passed": srm_p_value >= srm_alpha,
            },
            {
                "check": "missing_outcome_rate_gap",
                "value": missing_gap,
                "threshold": max_missing_rate_gap,
                "passed": missing_gap <= max_missing_rate_gap,
            },
            {
                "check": "holdout_contamination",
                "value": float(holdout["treatment_received_n"]),
                "threshold": 0.0,
                "passed": int(holdout["treatment_received_n"]) == 0,
            },
        ]
    )


def _validate_and_prepare(
    results: pd.DataFrame,
    design: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_results = sorted(REQUIRED_RESULT_COLUMNS - set(results.columns))
    missing_design = sorted(REQUIRED_DESIGN_COLUMNS - set(design.columns))
    if missing_results:
        raise ValueError(f"Results are missing columns: {missing_results}")
    if missing_design:
        raise ValueError(f"Design is missing columns: {missing_design}")
    if results["arm"].duplicated().any() or design["arm"].duplicated().any():
        raise ValueError("Each arm must appear exactly once.")

    clean_results = results.copy()
    count_columns = [
        "assigned_n",
        "outcome_observed_n",
        "outcomes",
        "targeted_n",
        "treatment_received_n",
    ]
    for column in count_columns:
        clean_results[column] = pd.to_numeric(
            clean_results[column], errors="raise"
        ).astype(int)
        if (clean_results[column] < 0).any():
            raise ValueError(f"{column} must not be negative.")
    if (clean_results["assigned_n"] <= 0).any():
        raise ValueError("assigned_n must be greater than zero.")
    if (clean_results["outcome_observed_n"] <= 0).any():
        raise ValueError("outcome_observed_n must be greater than zero.")
    if (clean_results["outcome_observed_n"] > clean_results["assigned_n"]).any():
        raise ValueError("outcome_observed_n must not exceed assigned_n.")
    if (clean_results["outcomes"] > clean_results["outcome_observed_n"]).any():
        raise ValueError("outcomes must not exceed outcome_observed_n.")
    if (clean_results["targeted_n"] > clean_results["assigned_n"]).any():
        raise ValueError("targeted_n must not exceed assigned_n.")
    if (clean_results["treatment_received_n"] > clean_results["targeted_n"]).any():
        raise ValueError("treatment_received_n must not exceed targeted_n.")
    if "total_campaign_cost" in clean_results:
        clean_results["total_campaign_cost"] = pd.to_numeric(
            clean_results["total_campaign_cost"], errors="raise"
        )
        if (clean_results["total_campaign_cost"] < 0).any():
            raise ValueError("total_campaign_cost must not be negative.")

    clean_design = design.copy()
    clean_design["sample_size"] = pd.to_numeric(
        clean_design["sample_size"], errors="raise"
    ).astype(int)
    if (clean_design["sample_size"] <= 0).any():
        raise ValueError("sample_size in the design must be greater than zero.")
    return clean_results, clean_design


def _get_arm(frame: pd.DataFrame, arm: str) -> pd.Series:
    selected = frame[frame["arm"] == arm]
    if len(selected) != 1:
        raise ValueError(f"Exactly one row is required for arm {arm}.")
    return selected.iloc[0]
