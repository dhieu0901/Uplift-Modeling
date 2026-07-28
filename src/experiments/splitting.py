from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.criteo import CriteoDataset


@dataclass(frozen=True)
class UpliftSplit:
    """One treatment/outcome-stratified partition of an uplift dataset."""

    X: pd.DataFrame
    y: pd.Series
    treatment: pd.Series
    indices: np.ndarray


@dataclass(frozen=True)
class HonestUpliftSplits:
    """Train, validation, and locked-test partitions."""

    train: UpliftSplit
    validation: UpliftSplit
    test: UpliftSplit


def split_train_validation_test(
    dataset: CriteoDataset,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    random_state: int = 42,
) -> HonestUpliftSplits:
    """Create an honest split while preserving treatment/outcome strata.

    Training data fit candidate learners and evaluation nuisance models.
    Validation data select the learner, hyperparameters, and operating budget.
    The test partition must remain untouched until all choices are locked.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in the interval (0, 1).")
    if not 0.0 < validation_fraction < 1.0 - train_fraction:
        raise ValueError(
            "validation_fraction must be positive and smaller than the "
            "share remaining after training."
        )

    strata = _joint_strata(dataset.y, dataset.treatment)
    all_indices = np.arange(len(dataset.X))
    train_indices, remainder_indices = train_test_split(
        all_indices,
        train_size=train_fraction,
        random_state=random_state,
        stratify=strata,
    )

    validation_share = validation_fraction / (1.0 - train_fraction)
    remainder_strata = strata.iloc[remainder_indices]
    validation_indices, test_indices = train_test_split(
        remainder_indices,
        train_size=validation_share,
        random_state=random_state + 1,
        stratify=remainder_strata,
    )

    return HonestUpliftSplits(
        train=_subset_split(dataset, train_indices),
        validation=_subset_split(dataset, validation_indices),
        test=_subset_split(dataset, test_indices),
    )


def _joint_strata(y: pd.Series, treatment: pd.Series) -> pd.Series:
    return treatment.astype(str) + "_" + y.astype(str)


def _subset_split(
    dataset: CriteoDataset,
    indices: np.ndarray,
) -> UpliftSplit:
    return UpliftSplit(
        X=dataset.X.iloc[indices].reset_index(drop=True),
        y=dataset.y.iloc[indices].reset_index(drop=True),
        treatment=dataset.treatment.iloc[indices].reset_index(drop=True),
        indices=np.asarray(indices, dtype=int),
    )
