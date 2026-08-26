from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def make_classifier(random_state: int = 42, **kwargs: Any):
    """Create the default probabilistic classifier.

    LightGBM is preferred for the project. The sklearn fallback keeps the repo
    runnable in a fresh environment where LightGBM has not been installed yet.

    ``n_jobs=-1`` costs nothing measurable in reproducibility here: on this
    project's data and parameters, predictions are bit-identical across 1, 8,
    and all cores. That is a measurement on one machine and one LightGBM build,
    not a guarantee - LightGBM only promises stable sums under
    ``deterministic=true``, which this project does not set because turning it
    on would move every published number. See ``docs/determinism.md``.
    """
    try:
        from lightgbm import LGBMClassifier

        params = {
            "n_estimators": 250,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_jobs": -1,
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
            "n_jobs": -1,
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


class EstimatorConstructor(Protocol):
    """Takes the run's seed and returns a fresh, unfitted estimator.

    The seed is declared optional because every family gives it a default, and
    one caller depends on that: ``registry._scaled_logistic_regression`` builds
    the linear family's classifier outside any seeded run. It can, because that
    estimator is ``lbfgs`` logistic regression, which ignores the seed. Writing
    the parameter as required would describe this field more narrowly than any
    implementation of it, which is how a type stops being documentation.
    """

    def __call__(self, random_state: int = ..., /) -> Any: ...


@dataclass(frozen=True)
class BaseLearnerFamily:
    """The estimator pair a meta-learner is built on.

    A meta-learner such as the S-learner or the DR-learner is a recipe for
    turning ordinary supervised estimators into an effect estimate. The recipe
    and the estimator it runs on are separate choices. Until this existed the
    project varied only the recipe, so every ranking it produced was a claim
    about the recipes *at one estimator* rather than about the recipes
    themselves.

    Both constructors take a seed and return a fresh estimator, because the
    learners build their components inside ``fit`` where the run's seed is
    known. Handing them a pre-built estimator would collapse the distinct seeds
    a multi-component learner gives its parts.
    """

    name: str
    classifier: EstimatorConstructor
    regressor: EstimatorConstructor


def gradient_boosting_family() -> BaseLearnerFamily:
    """Boosted trees: the estimator every published result in this repo used.

    Named for the method rather than the package because ``make_classifier``
    falls back to sklearn's histogram booster where LightGBM is missing.
    """
    return BaseLearnerFamily(
        name="gradient_boosting",
        classifier=make_classifier,
        regressor=make_regressor,
    )


def linear_family() -> BaseLearnerFamily:
    """Scaled linear models: high bias, low variance, the far end of the range.

    Scaling is not cosmetic here. Both estimators are penalised, and a penalty
    is only comparable across coefficients when the features share a scale.
    """

    def classifier(random_state: int = 42):
        return make_pipeline(
            StandardScaler(),
            # The default 100 iterations do not converge on samples this size,
            # and a warning is not a result, so give lbfgs room to finish.
            LogisticRegression(max_iter=1000, random_state=random_state),
        )

    def regressor(random_state: int = 42):
        del random_state  # Ridge has a closed-form solution.
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    return BaseLearnerFamily(
        name="linear",
        classifier=classifier,
        regressor=regressor,
    )


def forest_family() -> BaseLearnerFamily:
    """Bagged deep trees: tree-based like boosting, but a different bias.

    Boosting and bagging disagree about where accuracy comes from. Boosting
    grows many shallow trees that each correct the last; a forest grows deep
    independent trees and averages away their variance. Both are here so that a
    ranking which survives is known to survive the split between them, not just
    a change of hyperparameter inside one of them.

    ``min_samples_leaf`` is what keeps the trees from growing to purity on
    samples this size, which would cost hours and return a memorised training
    set. ``max_features`` is set on both estimators rather than left to the
    defaults, which differ: the classifier would take the square root of the
    feature count and the regressor would take all of them, so a forest
    S-learner and a forest X-learner would not be the same kind of forest.
    """

    def classifier(random_state: int = 42):
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=50,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )

    def regressor(random_state: int = 42):
        return RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=50,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )

    return BaseLearnerFamily(
        name="forest",
        classifier=classifier,
        regressor=regressor,
    )


#: Every family the base-learner comparison can draw from, by name.
BASE_LEARNER_FAMILIES: dict[str, Callable[[], BaseLearnerFamily]] = {
    "gradient_boosting": gradient_boosting_family,
    "linear": linear_family,
    "forest": forest_family,
}


def resolve_base_family(family: BaseLearnerFamily | str | None) -> BaseLearnerFamily:
    """Return the family to build a learner's components from.

    ``None`` returns the boosted-tree family, so a learner given no family
    behaves exactly as it did before families existed. Every number already
    published by this project was produced on that path.
    """
    if family is None:
        return gradient_boosting_family()
    if isinstance(family, BaseLearnerFamily):
        return family
    if family not in BASE_LEARNER_FAMILIES:
        known = ", ".join(sorted(BASE_LEARNER_FAMILIES))
        raise ValueError(f"Unknown base learner family {family!r}. Known: {known}.")
    return BASE_LEARNER_FAMILIES[family]()


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
