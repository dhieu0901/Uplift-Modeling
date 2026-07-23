import numpy as np

from src.evaluation.bootstrap import bootstrap_uplift_uncertainty


def test_bootstrap_returns_policy_and_business_intervals():
    treatment = np.tile([0, 1], 100)
    y = np.tile([0, 1, 1, 0, 0, 1, 0, 0], 25)
    base_score = np.linspace(1.0, 0.0, len(y))
    scores = {
        "random": np.random.default_rng(1).random(len(y)),
        "response_model": base_score,
        "s_learner": base_score + treatment * 0.01,
    }

    result = bootstrap_uplift_uncertainty(
        y,
        treatment,
        scores,
        champion="s_learner",
        fractions=(0.20,),
        n_bootstraps=20,
        random_state=7,
    )

    assert set(result.policy_metrics["policy"]) == set(scores)
    assert "difference_ci_lower" in result.policy_metrics.columns
    assert result.business_gains["budget_pct"].tolist() == [20.0]
    assert (
        result.business_gains["gain_vs_response_ci_lower"]
        <= result.business_gains["gain_vs_response_ci_upper"]
    ).all()
