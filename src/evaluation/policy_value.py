from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.uplift import budget_policy_table


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


def cost_benefit_policy_table(
    y: Sequence[float],
    treatment: Sequence[int],
    scores: Mapping[str, Sequence[float]],
    outcome_value: float,
    treatment_cost: float,
    fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """Evaluate the net value of ranking policies at each budget."""
    base_table = budget_policy_table(
        y,
        treatment,
        scores,
        fractions=fractions,
    )
    return monetize_policy_table(base_table, outcome_value, treatment_cost)


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


def select_best_campaign(
    policy_table: pd.DataFrame,
    allow_no_campaign: bool = True,
) -> pd.Series:
    """Select the policy/budget pair with the highest net value."""
    if "net_value" not in policy_table:
        raise ValueError("Policy table must contain a net_value column.")
    if policy_table.empty:
        raise ValueError("Policy table must not be empty.")

    valid = policy_table[np.isfinite(policy_table["net_value"])].copy()
    if valid.empty:
        raise ValueError("Policy table has no finite net_value values.")
    best = valid.loc[valid["net_value"].idxmax()].copy()
    if not allow_no_campaign or float(best["net_value"]) > 0:
        return best

    no_campaign = {column: np.nan for column in policy_table.columns}
    no_campaign.update(
        {
            "policy": "no_campaign",
            "budget_pct": 0.0,
            "n_targeted": 0,
            "incremental_outcome": 0.0,
            "gross_value": 0.0,
            "campaign_cost": 0.0,
            "net_value": 0.0,
            "net_value_per_1k_targeted": 0.0,
            "profitable": False,
        }
    )
    return pd.Series(no_campaign)


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
