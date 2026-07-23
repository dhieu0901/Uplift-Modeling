import numpy as np
import pandas as pd

from src.evaluation.policy_value import (
    monetize_policy_table,
    select_best_campaign,
    target_by_expected_value,
    uplift_score_threshold,
)


def test_uplift_score_threshold_and_target_policy():
    assert uplift_score_threshold(outcome_value=10.0, treatment_cost=2.0) == 0.2

    policy = target_by_expected_value(
        [0.1, 0.2, 0.3], outcome_value=10.0, treatment_cost=2.0
    )

    np.testing.assert_array_equal(policy, [False, False, True])


def test_monetize_policy_table_calculates_break_even_and_net_value():
    base = pd.DataFrame(
        [
            {
                "policy": "uplift",
                "budget_pct": 10.0,
                "n_targeted": 100,
                "incremental_outcome": 20.0,
            }
        ]
    )

    result = monetize_policy_table(
        base, outcome_value=5.0, treatment_cost=0.5
    ).iloc[0]

    assert result["gross_value"] == 100.0
    assert result["campaign_cost"] == 50.0
    assert result["net_value"] == 50.0
    assert result["net_value_per_1k_targeted"] == 500.0
    assert result["break_even_cost_per_target"] == 1.0
    assert result["minimum_value_to_cost_ratio"] == 5.0
    assert bool(result["profitable"])


def test_select_best_campaign_includes_no_campaign_option():
    policies = pd.DataFrame(
        [
            {"policy": "a", "budget_pct": 5.0, "net_value": -10.0},
            {"policy": "b", "budget_pct": 10.0, "net_value": -2.0},
        ]
    )

    best = select_best_campaign(policies)

    assert best["policy"] == "no_campaign"
    assert best["budget_pct"] == 0.0
    assert best["net_value"] == 0.0


def test_select_best_campaign_returns_highest_positive_value():
    policies = pd.DataFrame(
        [
            {"policy": "a", "budget_pct": 5.0, "net_value": 10.0},
            {"policy": "b", "budget_pct": 10.0, "net_value": 20.0},
        ]
    )

    best = select_best_campaign(policies)

    assert best["policy"] == "b"
    assert best["budget_pct"] == 10.0
