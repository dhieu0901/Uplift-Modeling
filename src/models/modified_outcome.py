from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseLearnerFamily, resolve_base_family


@dataclass
class ModifiedOutcomeModel:
    """Transformed-outcome learner for randomized treatment.

    The transformation appears in Athey and Imbens, *Recursive Partitioning for
    Heterogeneous Causal Effects*, PNAS 113(27):7353-7360.

    The regression target is Y(T-e)/(e(1-e)). In a randomized experiment where
    propensity e is independent of X, the conditional expectation of this target
    is the treatment effect.

    Ridge rather than gradient boosting is deliberate. At e = 0.85 the target
    takes only three values: -6.67 for a responding control, +1.18 for a
    responding treated row, and exactly 0 for the ~95% who do not respond. A
    flexible learner fits those spikes; a linear model constrained by an L2
    penalty cannot.
    """

    model: Any = None
    base_family: BaseLearnerFamily | str | None = None
    alpha: float = 1.0
    propensity_: float | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
        random_state: int = 42,
    ) -> ModifiedOutcomeModel:
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
            if self.base_family is None:
                # Ridge is deliberate here, for the reason in the class
                # docstring, and stays the default so the published
                # transformed-outcome numbers reproduce unchanged. Naming a
                # family overrides it, which is how the base-learner comparison
                # puts that reasoning to the test.
                self.model = make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=self.alpha),
                )
            else:
                self.model = resolve_base_family(self.base_family).regressor(random_state)
        self.model.fit(X, transformed_outcome)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ModifiedOutcomeModel has not been fitted.")
        return np.asarray(self.model.predict(X), dtype=float)
