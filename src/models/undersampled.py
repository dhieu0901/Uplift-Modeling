from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.imbalance import stratified_negative_undersample
from src.models.cvt_learner import CVTLearner
from src.models.t_learner import TLearner


@dataclass
class UndersampledTLearner:
    """T-learner trained with treatment-stratified negative undersampling."""

    factor: float
    learner: TLearner | None = None
    sampled_size_: int | None = None
    negative_keep_rates_: tuple[float, float] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
        random_state: int = 42,
    ) -> UndersampledTLearner:
        sampled = stratified_negative_undersample(
            X,
            y,
            treatment,
            factor=self.factor,
            random_state=random_state,
        )
        if self.learner is None:
            self.learner = TLearner()
        self.learner.fit(
            sampled.X,
            sampled.y,
            sampled.treatment,
            random_state=random_state,
        )
        self.sampled_size_ = sampled.sampled_size
        self.negative_keep_rates_ = sampled.negative_keep_rates
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if (
            self.learner is None
            or self.sampled_size_ is None
            or self.negative_keep_rates_ is None
        ):
            raise RuntimeError("UndersampledTLearner has not been fitted.")
        sampled_mu0 = np.asarray(self.learner.predict_mu0(X), dtype=float)
        sampled_mu1 = np.asarray(self.learner.predict_mu1(X), dtype=float)
        mu0 = _restore_original_probability(
            sampled_mu0,
            self.negative_keep_rates_[0],
        )
        mu1 = _restore_original_probability(
            sampled_mu1,
            self.negative_keep_rates_[1],
        )
        return mu1 - mu0


@dataclass
class UndersampledCVTLearner:
    """CVT learner trained with treatment-stratified negative undersampling.

    Unlike :class:`UndersampledTLearner`, this learner returns a score on the
    *undersampled* outcome scale and cannot be corrected back to the original
    scale. CVT with propensity rebalancing estimates the single quantity
    ``mu1_s(X) - mu0_s(X)``, whereas inverting case-control sampling requires
    each arm's probability separately: ``mu_t = r_t * mu_ts / (1 - mu_ts +
    r_t * mu_ts)`` is nonlinear, so the difference of corrected probabilities
    is not recoverable from the corrected difference.

    A previous version divided the score by ``factor``. That is a positive
    constant rescaling, so it left every ranking, every fixed-budget policy,
    and every reported number unchanged while implying a calibration that had
    not been performed. The division is gone and the scale is now documented
    instead: this score is valid for ranking and fixed-budget targeting, and
    must not be read as an absolute probability difference. Use
    :class:`UndersampledTLearner` when absolute magnitude matters.
    """

    factor: float
    learner: CVTLearner | None = None
    sampled_size_: int | None = None
    negative_keep_rates_: tuple[float, float] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
        random_state: int = 42,
    ) -> UndersampledCVTLearner:
        sampled = stratified_negative_undersample(
            X,
            y,
            treatment,
            factor=self.factor,
            random_state=random_state,
        )
        if self.learner is None:
            self.learner = CVTLearner()
        self.learner.fit(
            sampled.X,
            sampled.y,
            sampled.treatment,
            random_state=random_state,
        )
        self.sampled_size_ = sampled.sampled_size
        self.negative_keep_rates_ = sampled.negative_keep_rates
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Return a rank-valid uplift score on the undersampled scale."""
        if self.learner is None or self.sampled_size_ is None:
            raise RuntimeError("UndersampledCVTLearner has not been fitted.")
        return np.asarray(self.learner.predict_uplift(X), dtype=float)


def _restore_original_probability(
    sampled_probability: np.ndarray,
    negative_keep_rate: float,
) -> np.ndarray:
    """Invert case-control undersampling using the observed keep rate."""
    sampled_probability = np.clip(
        np.asarray(sampled_probability, dtype=float),
        0.0,
        1.0,
    )
    numerator = negative_keep_rate * sampled_probability
    denominator = (
        1.0 - sampled_probability + numerator
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
