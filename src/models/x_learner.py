from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.base import make_classifier, make_regressor, predict_positive_proba


@dataclass
class XLearner:
    """X-learner following Künzel et al. (2019).

    Stage 1 estimates potential outcome models. Stage 2 imputes treatment
    effects and learns them separately in treated/control groups. Predictions
    are combined by a propensity weight. For randomized experiments with a
    constant treatment ratio, the empirical treatment rate is a sensible first
    weighting choice.
    """

    treated_outcome_model: object | None = None
    control_outcome_model: object | None = None
    treated_effect_model: object | None = None
    control_effect_model: object | None = None
    propensity_: float | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
        random_state: int = 42,
    ) -> "XLearner":
        if self.treated_outcome_model is None:
            self.treated_outcome_model = make_classifier(random_state=random_state)
        if self.control_outcome_model is None:
            self.control_outcome_model = make_classifier(random_state=random_state + 1)
        if self.treated_effect_model is None:
            self.treated_effect_model = make_regressor(random_state=random_state + 2)
        if self.control_effect_model is None:
            self.control_effect_model = make_regressor(random_state=random_state + 3)

        treated_mask = treatment.to_numpy() == 1
        control_mask = ~treated_mask
        self.propensity_ = float(np.mean(treated_mask))

        X_t = X.loc[treated_mask]
        y_t = y.loc[treated_mask]
        X_c = X.loc[control_mask]
        y_c = y.loc[control_mask]

        self.treated_outcome_model.fit(X_t, y_t)
        self.control_outcome_model.fit(X_c, y_c)

        mu0_on_treated = predict_positive_proba(self.control_outcome_model, X_t)
        mu1_on_control = predict_positive_proba(self.treated_outcome_model, X_c)

        d_treated = y_t.to_numpy(dtype=float) - mu0_on_treated
        d_control = mu1_on_control - y_c.to_numpy(dtype=float)

        self.treated_effect_model.fit(X_t, d_treated)
        self.control_effect_model.fit(X_c, d_control)
        return self

    def predict_uplift(self, X: pd.DataFrame):
        if (
            self.treated_effect_model is None
            or self.control_effect_model is None
            or self.propensity_ is None
        ):
            raise RuntimeError("XLearner has not been fitted.")

        tau_treated = self.treated_effect_model.predict(X)
        tau_control = self.control_effect_model.predict(X)

        # Künzel et al. combine tau_0 and tau_1 with g(x). With a constant
        # propensity in an RCT, use the empirical treatment probability.
        g = self.propensity_
        return g * tau_control + (1.0 - g) * tau_treated
