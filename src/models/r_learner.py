from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import (
    BaseLearnerFamily,
    fit_with_sample_weight,
    resolve_base_family,
)
from src.models.cross_fitting import cross_fitted_potential_outcomes


@dataclass
class RLearner:
    """Cross-fitted R-learner for heterogeneous treatment effects.

    Follows Nie and Wager, *Quasi-Oracle Estimation of Heterogeneous Treatment
    Effects*, Biometrika 108(2):299-319, which builds on Robinson's partially
    linear decomposition ``Y - m(X) = (T - e) * tau(X) + noise``. Dividing by
    the treatment residual turns that into a regression target, and weighting
    by its square cancels the blow-up when ``T`` lands close to ``e``.

    In the randomized Criteo experiment the treatment propensity is constant.
    Cross-fitted arm-specific outcome models estimate the marginal outcome
    nuisance m(x), and the final regression minimizes the R-loss.
    """

    effect_model: Any = None
    outcome_model_factory: Callable[[int], Any] | None = None
    base_family: BaseLearnerFamily | str | None = None
    n_splits: int = 5
    propensity_: float | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
        random_state: int = 42,
    ) -> RLearner:
        treatment_arr = treatment.to_numpy(dtype=float)
        self.propensity_ = float(treatment_arr.mean())
        if not 0.0 < self.propensity_ < 1.0:
            raise ValueError("R-learner requires both treatment arms.")

        family = resolve_base_family(self.base_family)
        outcome_model_factory = self.outcome_model_factory or family.classifier
        mu0, mu1 = cross_fitted_potential_outcomes(
            X,
            y,
            treatment,
            model_factory=outcome_model_factory,
            n_splits=self.n_splits,
            random_state=random_state,
        )
        marginal_outcome = (
            self.propensity_ * mu1 + (1.0 - self.propensity_) * mu0
        )
        treatment_residual = treatment_arr - self.propensity_
        pseudo_outcome = (
            y.to_numpy(dtype=float) - marginal_outcome
        ) / treatment_residual
        sample_weight = treatment_residual**2

        if self.effect_model is None:
            self.effect_model = family.regressor(random_state + 1000)
        # The R-loss weights are not optional, so the fit has to go through the
        # adapter: a scaled linear effect model is a pipeline, and a pipeline
        # only takes a sample weight addressed to its final step.
        fit_with_sample_weight(self.effect_model, X, pseudo_outcome, sample_weight)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if self.effect_model is None:
            raise RuntimeError("RLearner has not been fitted.")
        return np.asarray(self.effect_model.predict(X), dtype=float)
