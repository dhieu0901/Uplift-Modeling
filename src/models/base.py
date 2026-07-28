from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline


def make_classifier(random_state: int = 42, **kwargs: Any):
    """Create the default probabilistic classifier.

    LightGBM is preferred for the project. The sklearn fallback keeps the repo
    runnable in a fresh environment where LightGBM has not been installed yet.
    """
    try:
        from lightgbm import LGBMClassifier

        params = {
            "n_estimators": 250,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_jobs": 1,
            "random_state": random_state,
            "verbosity": -1,
        }
        params.update(kwargs)
        return LGBMClassifier(**params)
    except ImportError:
        params = {
            "max_iter": 250,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.01,
            "random_state": random_state,
        }
        params.update({k: v for k, v in kwargs.items() if k in params})
        return HistGradientBoostingClassifier(**params)


def make_regressor(random_state: int = 42, **kwargs: Any):
    """Create the default regressor for second-stage treatment effects."""
    try:
        from lightgbm import LGBMRegressor

        params = {
            "n_estimators": 250,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_jobs": 1,
            "random_state": random_state,
            "verbosity": -1,
        }
        params.update(kwargs)
        return LGBMRegressor(**params)
    except ImportError:
        params = {
            "max_iter": 250,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.01,
            "random_state": random_state,
        }
        params.update({k: v for k, v in kwargs.items() if k in params})
        return HistGradientBoostingRegressor(**params)


def predict_positive_proba(model, X) -> np.ndarray:
    """Return P(y=1|X) for classifiers with a sklearn-like API."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba
    prediction = model.predict(X)
    return np.asarray(prediction, dtype=float)


def fit_with_sample_weight(model, X, y, sample_weight):
    """Fit estimators and sklearn pipelines with a common weight interface."""
    if sample_weight is None:
        return model.fit(X, y)
    if isinstance(model, Pipeline):
        final_step_name = model.steps[-1][0]
        return model.fit(
            X,
            y,
            **{f"{final_step_name}__sample_weight": sample_weight},
        )
    return model.fit(X, y, sample_weight=sample_weight)
