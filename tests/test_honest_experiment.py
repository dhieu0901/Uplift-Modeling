import numpy as np
import pandas as pd
import pytest

from src.data.semisynthetic import SemiSyntheticUpliftDataset
from src.experiments.honest_uplift import (
    _model_seed,
    evaluate_locked_policies,
    multiplicity_adjusted_bounds,
    select_validation_champion,
)
from src.experiments.splitting import UpliftSplit
from src.models.registry import select_model_factories


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


def _contrast_row(policy: str, difference: float, half_width: float) -> dict:
    return {
        "policy": policy,
        "budget_pct": 5.0,
        "difference": difference,
        "ci_lower": difference - half_width,
        "ci_upper": difference + half_width,
    }


def test_adjusting_for_the_candidate_count_pulls_every_bound_toward_zero():
    contrasts = pd.DataFrame(
        [
            _contrast_row("model_a", 100.0, 40.0),
            _contrast_row("model_b", 80.0, 40.0),
            _contrast_row("model_c", 60.0, 40.0),
        ]
    )

    adjusted = multiplicity_adjusted_bounds(
        contrasts,
        candidate_policies=["model_a", "model_b", "model_c"],
        primary_budget=0.05,
    )

    assert set(adjusted["n_candidates"]) == {3}
    merged = contrasts.merge(adjusted, on="policy")
    assert (merged["ci_lower_adjusted"] < merged["ci_lower"]).all()
    # Equal standard errors, so the correction is a constant shift and the
    # ordering the selection rule reads is untouched.
    shift = merged["ci_lower"] - merged["ci_lower_adjusted"]
    np.testing.assert_allclose(shift, shift.iloc[0])


def test_a_single_candidate_needs_no_adjustment():
    contrasts = pd.DataFrame([_contrast_row("model_a", 100.0, 40.0)])

    adjusted = multiplicity_adjusted_bounds(
        contrasts,
        candidate_policies=["model_a"],
        primary_budget=0.05,
    )

    assert adjusted["adjusted_confidence_level"].iloc[0] == pytest.approx(0.95)
    assert adjusted["ci_lower_adjusted"].iloc[0] == pytest.approx(60.0)


def test_a_wide_interval_is_penalised_more_than_a_narrow_one():
    # This is why the correction is reported next to the selection rule rather
    # than used by it: the two candidates swap order under adjustment.
    contrasts = pd.DataFrame(
        [
            _contrast_row("wide", 300.0, 250.0),
            _contrast_row("narrow", 120.0, 80.0),
        ]
    )

    adjusted = multiplicity_adjusted_bounds(
        contrasts,
        candidate_policies=["wide", "narrow"],
        primary_budget=0.05,
    ).set_index("policy")

    assert contrasts.set_index("policy").loc["wide", "ci_lower"] > (
        contrasts.set_index("policy").loc["narrow", "ci_lower"]
    )
    assert (
        adjusted.loc["wide", "ci_lower_adjusted"]
        < adjusted.loc["narrow", "ci_lower_adjusted"]
    )


def test_model_seed_depends_only_on_the_base_seed_and_the_model_name():
    # Seeding by position in the factory dict meant that adding one candidate
    # silently reseeded every candidate after it, so two runs sharing a learner
    # were not comparable. Name-derived seeds remove that coupling.
    assert _model_seed(777, "s_learner") == _model_seed(777, "s_learner")
    assert _model_seed(777, "s_learner") != _model_seed(777, "t_learner")
    assert _model_seed(777, "s_learner") != _model_seed(778, "s_learner")
    assert 0 <= _model_seed(777, "s_learner") < 2**31 - 1


def test_locked_evaluation_scores_a_separate_sample_once(
    small_uplift_dataset: SemiSyntheticUpliftDataset,
):
    dataset = small_uplift_dataset.dataset
    fit_data = _slice_dataset(dataset, 0, 2000)
    test_data = _slice_dataset(dataset, 2000, 3000)
    factories = select_model_factories(["s_learner"])

    result = evaluate_locked_policies(
        fit_data,
        test_data,
        factories,
        budgets=(0.05, 0.20),
        random_state=4,
    )

    assert result.n_fit == 2000
    assert result.n_test == 1000
    assert set(result.scores) == {
        "response_model",
        "random_targeting",
        "s_learner",
    }
    assert all(score.shape == (1000,) for score in result.scores.values())
    assert 0.0 < result.propensity < 1.0
    # The paired contrast is measured against the incumbent, which therefore
    # cannot appear as its own comparison row.
    assert "response_model" not in set(result.contrasts["policy"])
    assert (result.contrasts["reference_policy"] == "response_model").all()
    assert set(result.policy_values["budget_pct"]) == {5.0, 20.0}
    assert np.isfinite(result.policy_values["incremental_outcome"]).all()


def test_locked_evaluation_requires_the_reference_policy(
    small_uplift_dataset: SemiSyntheticUpliftDataset,
):
    dataset = small_uplift_dataset.dataset
    data = _slice_dataset(dataset, 0, 500)
    factories = {"s_learner": select_model_factories(["s_learner"])["s_learner"]}

    with pytest.raises(ValueError, match="response_model"):
        evaluate_locked_policies(data, data, factories)


def _slice_dataset(dataset, start: int, stop: int) -> UpliftSplit:
    index = np.arange(start, stop)
    return UpliftSplit(
        X=dataset.X.iloc[index].reset_index(drop=True),
        y=dataset.y.iloc[index].reset_index(drop=True),
        treatment=dataset.treatment.iloc[index].reset_index(drop=True),
        indices=index,
    )
