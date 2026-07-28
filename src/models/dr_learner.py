from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.base import make_classifier, make_regressor
from src.models.cross_fitting import cross_fitted_potential_outcomes


@dataclass
class DRLearner:
    """Cross-fitted doubly robust learner for CATE prediction."""

    effect_model: object | None = None
    outcome_model_factory: Callable[[int], object] | None = None
    n_splits: int = 5
    propensity_: float | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
        random_state: int = 42,
    ) -> "DRLearner":
        treatment_arr = treatment.to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)
        self.propensity_ = float(treatment_arr.mean())
        if not 0.0 < self.propensity_ < 1.0:
            raise ValueError("DR-learner requires both treatment arms.")

        outcome_model_factory = self.outcome_model_factory or (
            lambda seed: make_classifier(random_state=seed)
        )
        mu0, mu1 = cross_fitted_potential_outcomes(
            X,
            y,
            treatment,
            model_factory=outcome_model_factory,
            n_splits=self.n_splits,
            random_state=random_state,
        )
        pseudo_outcome = (
            mu1
            - mu0
            + treatment_arr
            / self.propensity_
            * (y_arr - mu1)
            - (1.0 - treatment_arr)
            / (1.0 - self.propensity_)
            * (y_arr - mu0)
        )

        if self.effect_model is None:
            self.effect_model = make_regressor(random_state=random_state + 2000)
        self.effect_model.fit(X, pseudo_outcome)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if self.effect_model is None:
            raise RuntimeError("DRLearner has not been fitted.")
        return np.asarray(self.effect_model.predict(X), dtype=float)
