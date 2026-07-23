from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.criteo import CriteoDataset
from src.models.cvt_learner import CVTLearner
from src.models.modified_outcome import ModifiedOutcomeModel
from src.models.response_model import ResponseModel
from src.models.s_learner import SLearner
from src.models.t_learner import TLearner
from src.models.x_learner import XLearner


@dataclass(frozen=True)
class CriteoExperimentResult:
    """Predictions and metadata from one Criteo train/test run."""

    y_test: pd.Series
    treatment_test: pd.Series
    scores: dict[str, np.ndarray]
    timing_table: pd.DataFrame
    train_size: int
    test_size: int


def run_criteo_experiment(
    dataset: CriteoDataset,
    test_fraction: float = 0.30,
    random_state: int = 42,
    progress: Callable[[str], None] | None = None,
    policies: Sequence[str] | None = None,
) -> CriteoExperimentResult:
    """Create a stratified split, train models, and return test-set predictions."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in the interval (0, 1).")

    strata = dataset.treatment.astype(str) + "_" + dataset.y.astype(str)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        dataset.X,
        dataset.y,
        dataset.treatment,
        test_size=test_fraction,
        random_state=random_state,
        stratify=strata,
    )

    models = {
        "response_model": ResponseModel(),
        "s_learner": SLearner(),
        "t_learner": TLearner(),
        "x_learner": XLearner(),
        "cvt": CVTLearner(),
        "transformed_outcome": ModifiedOutcomeModel(),
    }
    if policies is not None:
        requested = set(policies) - {"random"}
        unknown = sorted(requested - set(models))
        if unknown:
            raise ValueError(f"Invalid policy: {unknown}")
        models = {name: model for name, model in models.items() if name in requested}

    rng = np.random.default_rng(random_state)
    scores: dict[str, np.ndarray] = {"random": rng.random(len(X_test))}
    timing_rows = []

    for name, model in models.items():
        if progress is not None:
            progress(name)
        started = perf_counter()
        model.fit(X_train, y_train, w_train, random_state=random_state)
        fit_seconds = perf_counter() - started
        if name == "response_model":
            score = model.predict_score(X_test)
        else:
            score = model.predict_uplift(X_test)
        scores[name] = np.asarray(score, dtype=float)
        timing_rows.append({"model": name, "fit_seconds": fit_seconds})

    return CriteoExperimentResult(
        y_test=y_test.reset_index(drop=True),
        treatment_test=w_test.reset_index(drop=True),
        scores=scores,
        timing_table=pd.DataFrame(timing_rows).sort_values("fit_seconds"),
        train_size=len(X_train),
        test_size=len(X_test),
    )
