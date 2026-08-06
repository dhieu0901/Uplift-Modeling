import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.imbalance import stratified_negative_undersample
from src.models.cvt_learner import CVTLearner
from src.models.t_learner import TLearner
from src.models.undersampled import UndersampledCVTLearner, UndersampledTLearner


def make_imbalanced_data():
    rows = []
    for treatment in (0, 1):
        for index in range(100):
            rows.append(
                {
                    "x": float(index),
                    "treatment": treatment,
                    "y": int(index < 10),
                }
            )
    frame = pd.DataFrame(rows)
    return frame[["x"]], frame["y"], frame["treatment"]


def test_stratified_undersampling_multiplies_arm_positive_rate():
    X, y, treatment = make_imbalanced_data()

    sampled = stratified_negative_undersample(
        X,
        y,
        treatment,
        factor=5.0,
        random_state=7,
    )

    assert sampled.sampled_size == 40
    np.testing.assert_allclose(sampled.negative_keep_rates, [1 / 9, 1 / 9])
    grouped_rates = pd.DataFrame(
        {"y": sampled.y, "treatment": sampled.treatment}
    ).groupby("treatment")["y"].mean()
    np.testing.assert_allclose(grouped_rates.to_numpy(), [0.5, 0.5])


def test_undersampled_t_learner_returns_corrected_finite_scores():
    X, y, treatment = make_imbalanced_data()
    learner = TLearner(
        treated_model=LogisticRegression(max_iter=500),
        control_model=LogisticRegression(max_iter=500),
    )
    model = UndersampledTLearner(factor=2.0, learner=learner)

    model.fit(X, y, treatment, random_state=3)
    score = model.predict_uplift(X)

    assert model.sampled_size_ == 100
    assert score.shape == (len(X),)
    assert np.isfinite(score).all()


def test_undersampled_cvt_reports_scores_on_the_sampled_scale():
    X, y, treatment = make_imbalanced_data()
    model = UndersampledCVTLearner(
        factor=5.0,
        learner=CVTLearner(
            model=make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=500),
            )
        ),
    )

    model.fit(X, y, treatment, random_state=3)
    score = model.predict_uplift(X)

    assert model.sampled_size_ == 40
    assert model.negative_keep_rates_ == (1 / 9, 1 / 9)
    assert np.isfinite(score).all()
    # The wrapper returns the inner CVT score untouched. Dividing by the
    # undersampling factor would be a positive constant rescaling: it changes
    # no ranking and no policy, so it would only imply a calibration that was
    # never performed. CVT estimates mu1 - mu0 as a single quantity, and
    # inverting case-control sampling needs each arm separately.
    np.testing.assert_allclose(score, model.learner.predict_uplift(X))
