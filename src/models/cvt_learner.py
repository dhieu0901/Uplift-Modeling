from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.base import (
    fit_with_sample_weight,
    make_classifier,
    predict_positive_proba,
)


@dataclass
class CVTLearner:
    """Class Variable Transformation with propensity balancing.

    The transformation is due to Jaskowski and Jaroszewicz, *Uplift Modeling for
    Clinical Trial Data*, ICML 2012 Workshop on Machine Learning for Clinical
    Data Analysis.

    Classical CVT uses the label Z = 1(Y = T) and uplift = 2P(Z=1|X)-1,
    which assumes balanced treatment and control groups. Criteo has a propensity
    of approximately 0.85, so the model uses inverse-propensity weights by default
    to create a pseudo-population with equally weighted treatment arms.
    """

    model: object | None = None
    rebalance_treatment: bool = True
    propensity_: float | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
        random_state: int = 42,
    ) -> CVTLearner:
        treatment_values = treatment.to_numpy(dtype=int)
        self.propensity_ = float(treatment_values.mean())
        if not 0.0 < self.propensity_ < 1.0:
            raise ValueError("CVT requires both treatment and control observations.")

        if self.model is None:
            self.model = make_classifier(random_state=random_state)

        transformed_class = (y.to_numpy(dtype=int) == treatment_values).astype(int)
        sample_weight = None
        if self.rebalance_treatment:
            sample_weight = np.where(
                treatment_values == 1,
                0.5 / self.propensity_,
                0.5 / (1.0 - self.propensity_),
            )
        fit_with_sample_weight(
            self.model,
            X,
            transformed_class,
            sample_weight,
        )
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CVTLearner has not been fitted.")
        transformed_probability = predict_positive_proba(self.model, X)
        return 2.0 * transformed_probability - 1.0
