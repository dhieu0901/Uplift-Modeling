import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from src.models.cvt_learner import CVTLearner
from src.models.modified_outcome import ModifiedOutcomeModel


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


def test_modified_outcome_returns_finite_uplift():
    X, y, treatment = make_toy_data()
    model = ModifiedOutcomeModel(model=Ridge(alpha=1.0)).fit(X, y, treatment)

    uplift = model.predict_uplift(X)

    assert model.propensity_ == 0.5
    assert uplift.shape == (len(X),)
    assert np.isfinite(uplift).all()
