import pandas as pd
import pytest

from src.evaluation.online_experiment import (
    analyze_online_experiment,
    binary_itt_comparison,
    sample_ratio_mismatch,
)


def make_results():
    return pd.DataFrame(
        [
            {
                "arm": "A",
                "assigned_n": 1000,
                "outcome_observed_n": 1000,
                "outcomes": 120,
                "targeted_n": 100,
                "treatment_received_n": 95,
            },
            {
                "arm": "B",
                "assigned_n": 1000,
                "outcome_observed_n": 1000,
                "outcomes": 100,
                "targeted_n": 100,
                "treatment_received_n": 95,
            },
            {
                "arm": "H",
                "assigned_n": 500,
                "outcome_observed_n": 500,
                "outcomes": 40,
                "targeted_n": 0,
                "treatment_received_n": 0,
            },
        ]
    )


def make_design():
    return pd.DataFrame(
        [
            {"arm": "A", "sample_size": 1000},
            {"arm": "B", "sample_size": 1000},
            {"arm": "H", "sample_size": 500},
        ]
    )


def test_binary_itt_comparison_uses_all_observed_users():
    result = binary_itt_comparison(make_results(), "A", "B")

    assert result["rate_a"] == pytest.approx(0.12)
    assert result["rate_b"] == pytest.approx(0.10)
    assert result["absolute_difference"] == pytest.approx(0.02)
    assert result["relative_lift"] == pytest.approx(0.20)


def test_sample_ratio_mismatch_passes_for_planned_allocation():
    _, statistic, p_value = sample_ratio_mismatch(make_results(), make_design())

    assert statistic == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)


def test_sample_ratio_mismatch_detects_skewed_allocation():
    results = make_results()
    results.loc[results["arm"] == "A", "assigned_n"] = 1200
    results.loc[results["arm"] == "B", "assigned_n"] = 800

    _, _, p_value = sample_ratio_mismatch(results, make_design())

    assert p_value < 0.01


def test_full_analysis_calculates_net_value_and_quality_checks():
    analysis = analyze_online_experiment(
        make_results(),
        make_design(),
        outcome_value=10.0,
        treatment_cost=1.0,
    )

    policy_a = analysis.policy_value.set_index("arm").loc["A"]
    assert policy_a["incremental_outcomes_vs_holdout"] == pytest.approx(40.0)
    assert policy_a["campaign_cost"] == pytest.approx(100.0)
    assert policy_a["net_value"] == pytest.approx(300.0)
    primary = analysis.comparisons.set_index("comparison").loc["primary_A_vs_B"]
    assert primary["net_value_difference_per_user"] == pytest.approx(0.2)
    assert analysis.quality_checks["passed"].all()


def test_analysis_rejects_treatment_received_above_targeted():
    results = make_results()
    results.loc[results["arm"] == "A", "treatment_received_n"] = 101

    with pytest.raises(ValueError, match="must not exceed targeted_n"):
        analyze_online_experiment(results, make_design())
