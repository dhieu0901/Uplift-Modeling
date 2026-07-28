from __future__ import annotations

from collections.abc import Callable, Iterable

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.cvt_learner import CVTLearner
from src.models.dr_learner import DRLearner
from src.models.modified_outcome import ModifiedOutcomeModel
from src.models.response_model import ResponseModel
from src.models.r_learner import RLearner
from src.models.s_learner import SLearner
from src.models.t_learner import TLearner
from src.models.undersampled import (
    UndersampledCVTLearner,
    UndersampledTLearner,
)
from src.models.x_learner import XLearner


ModelFactory = Callable[[], object]


def default_model_factories(
    crossfit_folds: int = 5,
) -> dict[str, ModelFactory]:
    """Return fresh factories for the standard response/uplift benchmark."""
    return {
        "response_model": ResponseModel,
        "s_learner": SLearner,
        "t_learner": TLearner,
        "x_learner": XLearner,
        "cvt": CVTLearner,
        "transformed_outcome": ModifiedOutcomeModel,
        "r_learner": lambda: RLearner(n_splits=crossfit_folds),
        "dr_learner": lambda: DRLearner(n_splits=crossfit_folds),
    }


def select_model_factories(
    names: Iterable[str],
    crossfit_folds: int = 5,
) -> dict[str, ModelFactory]:
    registry = default_model_factories(crossfit_folds=crossfit_folds)
    selected_names = list(dict.fromkeys(names))
    unknown = sorted(set(selected_names) - set(registry))
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    if "response_model" not in selected_names:
        selected_names.insert(0, "response_model")
    return {name: registry[name] for name in selected_names}


def rare_outcome_model_factories(
    factors: Iterable[float],
    families: Iterable[str] = ("t", "cvt"),
) -> dict[str, ModelFactory]:
    """Create logistic T/CVT candidates for stratified undersampling factors."""
    selected_families = tuple(dict.fromkeys(families))
    unknown = sorted(set(selected_families) - {"t", "cvt"})
    if unknown:
        raise ValueError(f"Unsupported rare-outcome model families: {unknown}")
    factories: dict[str, ModelFactory] = {}
    for factor_value in factors:
        factor = float(factor_value)
        label = f"{factor:g}".replace(".", "p")
        if "t" in selected_families:
            factories[f"undersampled_t_lr_k{label}"] = (
                lambda current_factor=factor: UndersampledTLearner(
                    factor=current_factor,
                    learner=TLearner(
                        treated_model=_scaled_logistic_regression(),
                        control_model=_scaled_logistic_regression(),
                    ),
                )
            )
        if "cvt" in selected_families:
            factories[f"undersampled_cvt_lr_k{label}"] = (
                lambda current_factor=factor: UndersampledCVTLearner(
                    factor=current_factor,
                    learner=CVTLearner(
                        model=_scaled_logistic_regression(),
                    ),
                )
            )
    return factories


def _scaled_logistic_regression():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
