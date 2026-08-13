import numpy as np
import pandas as pd
import pytest

from src.serving.campaign_policy import (
    CampaignPolicy,
    MeasuredBudget,
    measured_budgets_from_tables,
)


class _RankByFirstColumn:
    def predict_uplift(self, X):
        return X.iloc[:, 0].to_numpy(dtype=float)


def _policy(budgets=None) -> CampaignPolicy:
    return CampaignPolicy(
        model=_RankByFirstColumn(),
        model_name="s_learner",
        outcome="visit",
        feature_columns=["f0", "f1"],
        propensity=0.85,
        confidence_level=0.95,
        fit_sample="fit.parquet",
        fit_rows=1000,
        measured_sample="measured.parquet",
        measured_rows=4000,
        fitted_at="2026-08-12",
        model_seed=7,
        budgets=budgets
        or [
            MeasuredBudget(5.0, 0.004, 0.0035, 0.0045, 0.001, 0.0005, 0.0015),
            MeasuredBudget(20.0, 0.006, 0.0055, 0.0065, -0.0001, -0.0004, 0.0002),
        ],
    )


def _users(n: int) -> pd.DataFrame:
    return pd.DataFrame({"f0": np.arange(n, dtype=float), "f1": 0.0})


def test_selection_takes_exactly_the_budgeted_share_of_the_top_ranked():
    policy = _policy()

    mask = policy.select(_users(1000), 0.05)

    assert mask.sum() == 50
    # f0 ranks the users, so the selected set is the largest 50 values.
    assert set(np.flatnonzero(mask)) == set(range(950, 1000))


def test_an_unevaluated_budget_is_refused_rather_than_interpolated():
    policy = _policy()

    with pytest.raises(ValueError, match="never evaluated"):
        policy.expected_outcome(0.12, n_users=1000)


def test_the_projection_scales_the_measured_rate_to_the_campaign():
    policy = _policy()

    small = policy.expected_outcome(0.05, n_users=100_000)
    large = policy.expected_outcome(0.05, n_users=400_000)

    assert small["gain_vs_incumbent"] == pytest.approx(100.0)
    assert large["gain_vs_incumbent"] == pytest.approx(4 * 100.0)
    assert small["gain_excludes_zero"] is True


def test_a_budget_whose_interval_contains_zero_is_reported_as_such():
    policy = _policy()

    projection = policy.expected_outcome(0.20, n_users=100_000)

    assert projection["gain_excludes_zero"] is False


def test_missing_feature_columns_are_refused():
    policy = _policy()

    with pytest.raises(ValueError, match="missing feature columns"):
        policy.rank(pd.DataFrame({"f0": [1.0, 2.0]}))


def test_round_trip_through_disk_keeps_the_measurement(tmp_path):
    policy = _policy()
    path = policy.save(tmp_path / "policy.joblib")

    loaded = CampaignPolicy.load(path)

    assert loaded.measured_budgets() == [5.0, 20.0]
    assert loaded.model_seed == 7
    assert loaded.expected_outcome(0.05, 100_000)["gain_vs_incumbent"] == (
        pytest.approx(100.0)
    )


def test_rates_are_read_from_the_locked_tables_rather_than_recomputed():
    values = pd.DataFrame(
        [
            {
                "policy": "s_learner",
                "budget_pct": 5.0,
                "incremental_outcome_rate": 0.004573,
                "ci_lower_rate": 0.004332,
                "ci_upper_rate": 0.004814,
            }
        ]
    )
    contrasts = pd.DataFrame(
        [
            {
                "policy": "s_learner",
                "budget_pct": 5.0,
                "difference_rate": 0.001465,
                "standard_error_rate": 0.000129,
            }
        ]
    )

    budgets = measured_budgets_from_tables(
        values, contrasts, model_name="s_learner", confidence_level=0.95
    )

    assert len(budgets) == 1
    assert budgets[0].incremental_rate == pytest.approx(0.004573)
    # 0.001465 +/- 1.959964 * 0.000129
    assert budgets[0].gain_ci_lower_rate == pytest.approx(0.001212, abs=1e-6)
    assert budgets[0].gain_ci_upper_rate == pytest.approx(0.001718, abs=1e-6)


def test_a_model_that_was_never_measured_is_refused():
    values = pd.DataFrame([{"policy": "other", "budget_pct": 5.0}])

    with pytest.raises(ValueError, match="No policy values"):
        measured_budgets_from_tables(
            values, values, model_name="s_learner", confidence_level=0.95
        )
