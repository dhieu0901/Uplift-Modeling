import pytest

from src.evaluation.experiment_design import (
    buffered_sample_size,
    policy_outcome_rate,
    two_proportion_sample_size,
)


def test_policy_outcome_rate_adds_incremental_effect_per_population():
    result = policy_outcome_rate(
        no_campaign_rate=0.04,
        incremental_outcome=100.0,
        population_size=100_000,
    )

    assert result == pytest.approx(0.041)


def test_two_proportion_sample_size_matches_standard_formula():
    assert two_proportion_sample_size(0.10, 0.12, alpha=0.05, power=0.80) == 3841
    assert two_proportion_sample_size(0.12, 0.10, alpha=0.05, power=0.80) == 3841


def test_buffered_sample_size_rounds_up():
    assert buffered_sample_size(100, buffer_fraction=0.15) == 115


def test_two_proportion_sample_size_rejects_zero_effect():
    with pytest.raises(ValueError, match="effect size"):
        two_proportion_sample_size(0.10, 0.10)
