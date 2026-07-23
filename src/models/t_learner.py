from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.models.base import make_classifier, predict_positive_proba


@dataclass
class TLearner:
    """T-learner: separate outcome models for treated and control users."""

    treated_model: object | None = None
    control_model: object | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
        random_state: int = 42,
    ) -> "TLearner":
        if self.treated_model is None:
            self.treated_model = make_classifier(random_state=random_state)
        if self.control_model is None:
            self.control_model = make_classifier(random_state=random_state + 1)

        treated_mask = treatment.to_numpy() == 1
        control_mask = ~treated_mask

        self.treated_model.fit(X.loc[treated_mask], y.loc[treated_mask])
        self.control_model.fit(X.loc[control_mask], y.loc[control_mask])
        return self

    def predict_mu1(self, X: pd.DataFrame):
        if self.treated_model is None:
            raise RuntimeError("TLearner has not been fitted.")
        return predict_positive_proba(self.treated_model, X)

    def predict_mu0(self, X: pd.DataFrame):
        if self.control_model is None:
            raise RuntimeError("TLearner has not been fitted.")
        return predict_positive_proba(self.control_model, X)

    def predict_uplift(self, X: pd.DataFrame):
        return self.predict_mu1(X) - self.predict_mu0(X)
