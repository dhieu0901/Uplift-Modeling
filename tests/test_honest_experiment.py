import pandas as pd

from src.experiments.honest_uplift import select_validation_champion


def test_champion_is_selected_by_lower_bound_at_locked_budget_only():
    values = pd.DataFrame(
        [
            {
                "policy": "model_a",
                "budget_pct": 5.0,
                "difference": 100.0,
                "ci_lower": 10.0,
            },
            {
                "policy": "model_b",
                "budget_pct": 5.0,
                "difference": 80.0,
                "ci_lower": 20.0,
            },
            {
                "policy": "model_a",
                "budget_pct": 10.0,
                "difference": 500.0,
                "ci_lower": 400.0,
            },
        ]
    )

    champion = select_validation_champion(
        values,
        candidate_policies=["model_a", "model_b"],
        primary_budget=0.05,
    )

    assert champion == "model_b"
