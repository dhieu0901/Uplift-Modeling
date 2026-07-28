import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.cvt_learner import CVTLearner
from src.models.dr_learner import DRLearner
from src.models.modified_outcome import ModifiedOutcomeModel
from src.models.r_learner import RLearner


def make_toy_data():
    X = pd.DataFrame(
        {
            "x1": np.tile([0.0, 1.0, 2.0, 3.0], 10),
            "x2": np.repeat([0.0, 1.0], 20),
        }
    )
    treatment = pd.Series(np.tile([0, 1], 20))
    y = pd.Series(np.tile([0, 0, 1, 1, 0, 1, 0, 1], 5))
    return X, y, treatment


def test_cvt_predicts_scores_in_valid_range():
    X, y, treatment = make_toy_data()
    model = CVTLearner(model=LogisticRegression(max_iter=500)).fit(X, y, treatment)

    uplift = model.predict_uplift(X)

    assert model.propensity_ == 0.5
    assert uplift.shape == (len(X),)
    assert ((uplift >= -1.0) & (uplift <= 1.0)).all()


def test_cvt_supports_weighted_sklearn_pipeline():
    X, y, treatment = make_toy_data()
    model = CVTLearner(
        model=make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500),
        )
    ).fit(X, y, treatment)

    assert np.isfinite(model.predict_uplift(X)).all()


def test_modified_outcome_returns_finite_uplift():
    X, y, treatment = make_toy_data()
    model = ModifiedOutcomeModel(model=Ridge(alpha=1.0)).fit(X, y, treatment)

    uplift = model.predict_uplift(X)

    assert model.propensity_ == 0.5
    assert uplift.shape == (len(X),)
    assert np.isfinite(uplift).all()


def test_cross_fitted_r_and_dr_learners_return_finite_scores():
    X, y, treatment = make_toy_data()

    def outcome_factory(seed):
        return LogisticRegression(
            max_iter=500,
            random_state=seed,
        )

    models = [
        RLearner(
            effect_model=LinearRegression(),
            outcome_model_factory=outcome_factory,
            n_splits=2,
        ),
        DRLearner(
            effect_model=LinearRegression(),
            outcome_model_factory=outcome_factory,
            n_splits=2,
        ),
    ]

    for model in models:
        model.fit(X, y, treatment)
        uplift = model.predict_uplift(X)

        assert model.propensity_ == 0.5
        assert uplift.shape == (len(X),)
        assert np.isfinite(uplift).all()
