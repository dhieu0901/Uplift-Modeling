import numpy as np
import pandas as pd

from src.evaluation.policy_value import (
    aipw_policy_contrast_table,
    aipw_policy_value_table,
    doubly_robust_treatment_effect_scores,
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


def test_aipw_scores_equal_known_effect_with_perfect_nuisance_models():
    treatment = np.array([0, 1, 0, 1])
    mu0 = np.zeros(4)
    mu1 = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.where(treatment == 1, mu1, mu0)

    scores = doubly_robust_treatment_effect_scores(
        y,
        treatment,
        mu0,
        mu1,
        propensity=0.5,
    )

    np.testing.assert_allclose(scores, mu1 - mu0)


def test_aipw_policy_table_estimates_incremental_outcomes_and_interval():
    y = np.array([0, 1, 0, 1])
    treatment = np.array([0, 1, 0, 1])
    mu0 = np.zeros(4)
    mu1 = np.ones(4)
    scores = {"uplift": np.array([0.9, 0.8, 0.2, 0.1])}

    result = aipw_policy_value_table(
        y,
        treatment,
        scores,
        mu0,
        mu1,
        propensity=0.5,
        fractions=(0.5,),
        outcome_value=10.0,
        treatment_cost=1.0,
    ).iloc[0]

    assert result["n_targeted"] == 2
    assert result["incremental_outcome"] == 2.0
    assert result["ci_lower"] < result["incremental_outcome"]
    assert result["ci_upper"] > result["incremental_outcome"]
    assert result["net_value"] == 18.0


def test_aipw_policy_contrast_is_paired():
    treatment = np.array([0, 1, 0, 1])
    mu0 = np.zeros(4)
    mu1 = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.where(treatment == 1, mu1, mu0)
    scores = {
        "oracle": np.array([1.0, 0.0, 0.9, 0.1]),
        "response": np.array([0.0, 1.0, 0.1, 0.9]),
    }

    result = aipw_policy_contrast_table(
        y,
        treatment,
        scores,
        reference_policy="response",
        mu0=mu0,
        mu1=mu1,
        propensity=0.5,
        fractions=(0.5,),
    ).iloc[0]

    assert result["policy"] == "oracle"
    assert result["difference"] == 2.0
