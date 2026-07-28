from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModifiedOutcomeModel:
    """Modified-outcome learner from the Criteo benchmark.

    The regression target is Y(T-e)/(e(1-e)). In a randomized experiment where
    propensity e is independent of X, the conditional expectation of this target
    is the treatment effect.
    """

    model: object | None = None
    alpha: float = 1.0
    propensity_: float | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
        random_state: int = 42,
    ) -> "ModifiedOutcomeModel":
        del random_state  # Ridge is deterministic.
        treatment_values = treatment.to_numpy(dtype=float)
        self.propensity_ = float(treatment_values.mean())
        if not 0.0 < self.propensity_ < 1.0:
            raise ValueError(
                "Modified-outcome training requires treatment and control observations."
            )

        transformed_outcome = y.to_numpy(dtype=float) * (
            treatment_values - self.propensity_
        ) / (self.propensity_ * (1.0 - self.propensity_))

        if self.model is None:
            self.model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=self.alpha),
            )
        self.model.fit(X, transformed_outcome)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ModifiedOutcomeModel has not been fitted.")
        return np.asarray(self.model.predict(X), dtype=float)
