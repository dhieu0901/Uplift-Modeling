from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.semisynthetic import SemiSyntheticUpliftDataset
from src.experiments.honest_uplift import run_honest_uplift_experiment
from src.models.random_targeting import RandomTargetingModel
from src.models.registry import REFERENCE_POLICIES, select_model_factories


def test_random_targeting_is_reproducible_and_carries_no_information():
    features = pd.DataFrame(np.zeros((200, 3)))
    model = RandomTargetingModel().fit(
        features,
        pd.Series(np.zeros(200, dtype=int)),
        pd.Series(np.ones(200, dtype=int)),
        random_state=13,
    )

    first = model.predict_uplift(features)
    second = model.predict_uplift(features)

    assert np.array_equal(first, second)
    assert first.shape == (200,)
    assert ((first >= 0.0) & (first < 1.0)).all()
    # Identical rows still receive different scores: the score ranks users but
    # says nothing about any individual treatment effect.
    assert np.unique(first).size == 200


def test_selection_always_includes_both_reference_policies():
    factories = select_model_factories(["t_learner"])

    assert list(factories)[: len(REFERENCE_POLICIES)] == list(REFERENCE_POLICIES)
    assert "t_learner" in factories


def test_reference_policies_can_never_win_selection(
    small_uplift_dataset: SemiSyntheticUpliftDataset,
):
    factories = select_model_factories(["s_learner", "t_learner"])

    result = run_honest_uplift_experiment(
        small_uplift_dataset.dataset,
        model_factories=factories,
        budgets=(0.05, 0.20),
        primary_budget=0.05,
        random_state=5,
    )

    assert result.champion not in REFERENCE_POLICIES
    assert result.champion in {"s_learner", "t_learner"}
    # Both references are still measured on the locked test, so a reader can see
    # the incumbent and the floor next to the champion.
    evaluated = set(result.test_policy_values["policy"])
    assert set(REFERENCE_POLICIES) <= evaluated
    assert result.champion in evaluated
