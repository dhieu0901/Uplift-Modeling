import numpy as np
import pandas as pd

from src.data.semisynthetic import generate_semisynthetic_uplift
from src.evaluation.ground_truth import (
    ground_truth_cate_metrics,
    ground_truth_policy_table,
)


def test_semisynthetic_generator_is_reproducible_with_valid_probabilities():
    rng = np.random.default_rng(11)
    X = pd.DataFrame(
        rng.normal(size=(2000, 12)),
        columns=[f"f{index}" for index in range(12)],
    )

    first = generate_semisynthetic_uplift(X, random_state=9)
    second = generate_semisynthetic_uplift(X, random_state=9)

    np.testing.assert_allclose(first.mu0, second.mu0)
    np.testing.assert_allclose(first.mu1, second.mu1)
    np.testing.assert_array_equal(first.dataset.y, second.dataset.y)
    assert np.all((first.mu0 > 0.0) & (first.mu0 < 1.0))
    assert np.all((first.mu1 > 0.0) & (first.mu1 < 1.0))
    assert np.isclose(first.mu0.mean(), 0.05, atol=1e-4)
    assert np.std(first.true_cate) > 0.0


def test_ground_truth_metrics_reward_oracle_ranking():
    truth = np.array([0.4, 0.3, -0.2, -0.1])
    scores = {
        "oracle": truth,
        "reversed": -truth,
    }

    metrics = ground_truth_cate_metrics(truth, scores).set_index("policy")
    policies = ground_truth_policy_table(
        truth,
        scores,
        fractions=(0.5,),
    ).set_index("policy")

    assert metrics.loc["oracle", "pehe"] == 0.0
    assert policies.loc["oracle", "policy_regret"] == 0.0
    assert policies.loc["reversed", "policy_regret"] > 0.0
