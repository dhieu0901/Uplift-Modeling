from __future__ import annotations

import pytest

from src.models.registry import rare_outcome_model_factories


def test_rare_registry_can_lock_one_family() -> None:
    factories = rare_outcome_model_factories([1, 10], families=["cvt"])

    assert set(factories) == {
        "undersampled_cvt_lr_k1",
        "undersampled_cvt_lr_k10",
    }


def test_rare_registry_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        rare_outcome_model_factories([10], families=["unknown"])
