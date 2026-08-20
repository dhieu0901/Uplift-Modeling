from __future__ import annotations

from collections.abc import Callable, Iterable

from src.models.base import BaseLearnerFamily, linear_family
from src.models.cvt_learner import CVTLearner
from src.models.dr_learner import DRLearner
from src.models.modified_outcome import ModifiedOutcomeModel
from src.models.r_learner import RLearner
from src.models.random_targeting import RandomTargetingModel
from src.models.response_model import ResponseModel
from src.models.s_learner import SLearner
from src.models.t_learner import TLearner
from src.models.undersampled import (
    UndersampledCVTLearner,
    UndersampledTLearner,
)
from src.models.x_learner import XLearner

ModelFactory = Callable[[], object]

#: Policies that are always evaluated for context but can never win selection.
#: ``response_model`` is the incumbent the project must beat; ``random_targeting``
#: is the floor that shows how much of any reported gain is ranking skill rather
#: than the mere act of treating users.
REFERENCE_POLICIES = ("response_model", "random_targeting")


def default_model_factories(
    crossfit_folds: int = 5,
    base_family: BaseLearnerFamily | str | None = None,
) -> dict[str, ModelFactory]:
    """Return fresh factories for the standard response/uplift benchmark.

    ``base_family`` names the estimator the uplift candidates are built on, and
    ``None`` keeps the boosted trees every published result in this repo used.

    It deliberately does not reach the reference policies. ``response_model``
    stands for what the business already does, so it has to be one fixed bar
    that every family is measured against rather than a bar that moves with the
    family under test, and ``random_targeting`` builds no model at all.
    """
    return {
        "response_model": ResponseModel,
        "random_targeting": RandomTargetingModel,
        "s_learner": lambda: SLearner(base_family=base_family),
        "t_learner": lambda: TLearner(base_family=base_family),
        "x_learner": lambda: XLearner(base_family=base_family),
        "cvt": lambda: CVTLearner(base_family=base_family),
        "transformed_outcome": lambda: ModifiedOutcomeModel(base_family=base_family),
        "r_learner": lambda: RLearner(
            n_splits=crossfit_folds, base_family=base_family
        ),
        "dr_learner": lambda: DRLearner(
            n_splits=crossfit_folds, base_family=base_family
        ),
    }


def select_model_factories(
    names: Iterable[str],
    crossfit_folds: int = 5,
    base_family: BaseLearnerFamily | str | None = None,
) -> dict[str, ModelFactory]:
    registry = default_model_factories(
        crossfit_folds=crossfit_folds,
        base_family=base_family,
    )
    selected_names = list(dict.fromkeys(names))
    unknown = sorted(set(selected_names) - set(registry))
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    for reference in reversed(REFERENCE_POLICIES):
        if reference not in selected_names:
            selected_names.insert(0, reference)
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
    """The linear family's classifier, which is what these candidates want.

    Undersampling exists here because conversions are rare, and a penalised
    linear model is the estimator that survives a target that thin. Reading it
    off the family keeps one definition of what "scaled logistic" means.
    """
    return linear_family().classifier()
