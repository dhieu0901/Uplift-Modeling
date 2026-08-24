from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.criteo import CriteoDataset
from src.data.semisynthetic import SemiSyntheticUpliftDataset
from src.models.base import (
    BASE_LEARNER_FAMILIES,
    BaseLearnerFamily,
    forest_family,
    gradient_boosting_family,
    linear_family,
    make_classifier,
    make_regressor,
    resolve_base_family,
)
from src.models.cvt_learner import CVTLearner
from src.models.dr_learner import DRLearner
from src.models.modified_outcome import ModifiedOutcomeModel
from src.models.r_learner import RLearner
from src.models.registry import default_model_factories
from src.models.response_model import ResponseModel
from src.models.s_learner import SLearner
from src.models.t_learner import TLearner
from src.models.x_learner import XLearner

#: The seed the learners are fitted with throughout this module. The hand-built
#: comparisons below offset it exactly as the learners do internally.
SEED = 42

#: Cross-fitting folds for the two learners that need them. Three keeps the
#: test fast without changing what is being compared, since both sides of every
#: comparison use the same number.
FOLDS = 3


def _fit_uplift(model, dataset: CriteoDataset) -> np.ndarray:
    model.fit(dataset.X, dataset.y, dataset.treatment, random_state=SEED)
    return np.asarray(model.predict_uplift(dataset.X), dtype=float)


#: Tolerance for "these two paths compute the same thing".
#:
#: Not bit equality. LightGBM only promises reproducible sums under
#: ``deterministic=true``, which this project does not set, so two identical
#: fits can disagree in the last bit depending on how the machine schedules
#: threads. That was observed: the same commit passed on one CI interpreter and
#: failed on another with a largest absolute difference of 4.4e-16.
#:
#: This bound sits far above that noise and far below any real regression. A
#: wrong seed, a dropped sample weight, or a swapped estimator moves an uplift
#: score by 1e-3 or more, which is six orders of magnitude outside this window.
#: See ``docs/determinism.md``.
SAME_SCORES = {"rtol": 1e-9, "atol": 1e-12}


#: Each learner built two ways: left through the family default, right by hand.
#: The right-hand side names the estimators and the seeds the learners used
#: before base learner families existed, so a difference between the two sides
#: means a published number has moved. Values are callables because an
#: estimator instance cannot be fitted twice.
LEARNER_PAIRS: dict[str, Callable[[], tuple[object, object]]] = {
    "s_learner": lambda: (
        SLearner(),
        SLearner(model=make_classifier(random_state=SEED)),
    ),
    "t_learner": lambda: (
        TLearner(),
        TLearner(
            treated_model=make_classifier(random_state=SEED),
            control_model=make_classifier(random_state=SEED + 1),
        ),
    ),
    "x_learner": lambda: (
        XLearner(),
        XLearner(
            treated_outcome_model=make_classifier(random_state=SEED),
            control_outcome_model=make_classifier(random_state=SEED + 1),
            treated_effect_model=make_regressor(random_state=SEED + 2),
            control_effect_model=make_regressor(random_state=SEED + 3),
        ),
    ),
    "cvt": lambda: (
        CVTLearner(),
        CVTLearner(model=make_classifier(random_state=SEED)),
    ),
    "transformed_outcome": lambda: (
        ModifiedOutcomeModel(),
        ModifiedOutcomeModel(
            model=make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        ),
    ),
    "r_learner": lambda: (
        RLearner(n_splits=FOLDS),
        RLearner(
            n_splits=FOLDS,
            outcome_model_factory=lambda seed: make_classifier(random_state=seed),
            effect_model=make_regressor(random_state=SEED + 1000),
        ),
    ),
    "dr_learner": lambda: (
        DRLearner(n_splits=FOLDS),
        DRLearner(
            n_splits=FOLDS,
            outcome_model_factory=lambda seed: make_classifier(random_state=seed),
            effect_model=make_regressor(random_state=SEED + 2000),
        ),
    ),
}


@pytest.mark.parametrize("learner_name", sorted(LEARNER_PAIRS))
def test_default_learner_matches_the_hand_built_boosted_learner(
    learner_name: str,
    small_uplift_dataset: SemiSyntheticUpliftDataset,
) -> None:
    """Naming no family has to leave every learner exactly as it was.

    Every result this repository publishes was produced on that path, so this
    is the guard that adding families did not silently rewrite them.
    """
    dataset = small_uplift_dataset.dataset
    through_family, hand_built = LEARNER_PAIRS[learner_name]()

    np.testing.assert_allclose(
        _fit_uplift(through_family, dataset),
        _fit_uplift(hand_built, dataset),
        **SAME_SCORES,
    )


def test_naming_the_boosted_family_is_the_same_as_naming_nothing(
    small_uplift_dataset: SemiSyntheticUpliftDataset,
) -> None:
    dataset = small_uplift_dataset.dataset

    np.testing.assert_allclose(
        _fit_uplift(SLearner(), dataset),
        _fit_uplift(SLearner(base_family="gradient_boosting"), dataset),
        **SAME_SCORES,
    )


def test_linear_family_reproduces_the_transformed_outcome_default(
    small_uplift_dataset: SemiSyntheticUpliftDataset,
) -> None:
    """The locked configuration is a mixture, and this is where it shows.

    Seven candidates default to boosted trees, but the transformed-outcome
    learner defaults to ridge for the reason its docstring gives. Ridge is what
    the linear family's regressor is, so that one candidate is unchanged when
    the comparison moves to the linear family, and only the other six move.
    """
    dataset = small_uplift_dataset.dataset

    np.testing.assert_allclose(
        _fit_uplift(ModifiedOutcomeModel(), dataset),
        _fit_uplift(ModifiedOutcomeModel(base_family="linear"), dataset),
    )


def test_boosted_family_overrides_the_transformed_outcome_default(
    small_uplift_dataset: SemiSyntheticUpliftDataset,
) -> None:
    """Naming the boosted family is not the same as naming nothing.

    The locked configuration is a mixture: six candidates on boosted trees and
    the transformed-outcome learner on ridge, chosen for the reason in its
    docstring. Asking for a uniform boosted column overrides that, which is the
    point of asking. This pins the divergence so that a report comparing the
    boosted column against the locked run knows to expect exactly one row to
    move, and knows which one.
    """
    dataset = small_uplift_dataset.dataset

    assert not np.allclose(
        _fit_uplift(ModifiedOutcomeModel(), dataset),
        _fit_uplift(ModifiedOutcomeModel(base_family="gradient_boosting"), dataset),
    )


@pytest.mark.parametrize("family_name", sorted(BASE_LEARNER_FAMILIES))
@pytest.mark.parametrize("learner_name", sorted(LEARNER_PAIRS))
def test_every_learner_fits_on_every_family(
    learner_name: str,
    family_name: str,
    small_uplift_dataset: SemiSyntheticUpliftDataset,
) -> None:
    dataset = small_uplift_dataset.dataset
    factories = default_model_factories(crossfit_folds=FOLDS, base_family=family_name)

    uplift = _fit_uplift(factories[learner_name](), dataset)

    assert uplift.shape == (len(dataset.X),)
    assert np.isfinite(uplift).all()


def test_families_rank_users_differently(
    small_uplift_dataset: SemiSyntheticUpliftDataset,
) -> None:
    """Without this the comparison would be measuring nothing.

    Three families that scored users identically would make a stable ranking
    across them meaningless, since it would follow from the estimators being
    the same rather than from the meta-learner being robust to them.
    """
    dataset = small_uplift_dataset.dataset
    scores = {
        name: _fit_uplift(SLearner(base_family=name), dataset)
        for name in BASE_LEARNER_FAMILIES
    }

    for left, right in (
        ("gradient_boosting", "linear"),
        ("gradient_boosting", "forest"),
        ("linear", "forest"),
    ):
        assert not np.allclose(scores[left], scores[right])


def test_reference_policies_do_not_move_with_the_family() -> None:
    """The incumbent has to be one fixed bar, not one bar per family.

    Response targeting stands for what the business already does. If it were
    rebuilt on whichever estimator is under test, a family would be compared
    against a version of the incumbent that does not exist, and the incremental
    outcomes could not be read across families.
    """
    for family_name in BASE_LEARNER_FAMILIES:
        factories = default_model_factories(base_family=family_name)
        response = factories["response_model"]()

        assert isinstance(response, ResponseModel)
        assert response.base_family is None


def test_resolve_base_family_defaults_to_boosted_trees() -> None:
    assert resolve_base_family(None).name == "gradient_boosting"
    assert resolve_base_family("linear").name == "linear"


def test_resolve_base_family_passes_an_instance_through() -> None:
    family = forest_family()

    assert resolve_base_family(family) is family


def test_resolve_base_family_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown base learner family"):
        resolve_base_family("catboost")


@pytest.mark.parametrize(
    "family_builder",
    [gradient_boosting_family, linear_family, forest_family],
)
def test_every_family_builds_both_estimator_kinds(
    family_builder: Callable[[], BaseLearnerFamily],
) -> None:
    family = family_builder()

    assert family.name in BASE_LEARNER_FAMILIES
    assert hasattr(family.classifier(SEED), "fit")
    assert hasattr(family.regressor(SEED), "fit")
