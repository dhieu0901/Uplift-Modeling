from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.models.base import make_classifier, predict_positive_proba


@dataclass
class ResponseModel:
    """Business-as-usual targeting model.

    By default this is trained on treated users only, matching the common
    practice of learning who responds among people who received the campaign.
    """

    model: object | None = None
    trained_on_treated_only: bool = True

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series | None = None,
        random_state: int = 42,
    ) -> "ResponseModel":
        if self.model is None:
            self.model = make_classifier(random_state=random_state)

        if self.trained_on_treated_only:
            if treatment is None:
                raise ValueError("treatment is required when trained_on_treated_only=True.")
            mask = treatment.to_numpy() == 1
            X_fit = X.loc[mask]
            y_fit = y.loc[mask]
        else:
            X_fit = X
            y_fit = y

        self.model.fit(X_fit, y_fit)
        return self

    def predict_score(self, X: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("ResponseModel has not been fitted.")
        return predict_positive_proba(self.model, X)
