from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.models.base import make_classifier, predict_positive_proba


@dataclass
class SLearner:
    """S-learner: one outcome model with treatment as a feature."""

    model: object | None = None
    treatment_column: str = "_treatment"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
        random_state: int = 42,
    ) -> "SLearner":
        if self.model is None:
            self.model = make_classifier(random_state=random_state)
        X_fit = self._with_treatment(X, treatment)
        self.model.fit(X_fit, y)
        return self

    def predict_uplift(self, X: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("SLearner has not been fitted.")
        treated = pd.Series(1, index=X.index)
        control = pd.Series(0, index=X.index)
        mu1 = predict_positive_proba(self.model, self._with_treatment(X, treated))
        mu0 = predict_positive_proba(self.model, self._with_treatment(X, control))
        return mu1 - mu0

    def _with_treatment(self, X: pd.DataFrame, treatment: pd.Series) -> pd.DataFrame:
        X_aug = X.copy()
        X_aug[self.treatment_column] = treatment.to_numpy()
        return X_aug
