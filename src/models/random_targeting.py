from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RandomTargetingModel:
    """Reference policy that spends the budget on a uniformly random subset.

    This is the floor that any targeting model must clear. Without it a reader
    cannot tell how much of the reported incremental outcome comes from ranking
    skill and how much comes from simply treating people at all: at a 5% budget
    a random policy already collects roughly 5% of the average treatment effect.

    The model is a reference, never a champion candidate. It carries a
    ``predict_uplift`` method only so that it flows through the same evaluation
    path as the real learners; the returned values rank users and carry no
    information about any individual effect.
    """

    random_state_: int | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
        random_state: int = 42,
    ) -> RandomTargetingModel:
        del X, y, treatment  # A random policy has nothing to learn.
        self.random_state_ = int(random_state)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Return reproducible uniform scores in [0, 1)."""
        if self.random_state_ is None:
            raise RuntimeError("RandomTargetingModel has not been fitted.")
        generator = np.random.default_rng(self.random_state_)
        return generator.random(len(X))
