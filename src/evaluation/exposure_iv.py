from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a normal-approximation interval."""

    value: float
    standard_error: float
    ci_lower: float
    ci_upper: float

    @classmethod
    def from_se(
        cls,
        value: float,
        standard_error: float,
        confidence_level: float = 0.95,
    ) -> Estimate:
        z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
        return cls(
            value=float(value),
            standard_error=float(standard_error),
            ci_lower=float(value - z * standard_error),
            ci_upper=float(value + z * standard_error),
        )


@dataclass(frozen=True)
class ExposureAnalysis:
    """Everything the exposure question needs, at one outcome."""

    outcome: str
    n: int
    rate_control: float
    rate_assigned_unexposed: float
    rate_assigned_exposed: float
    n_control: int
    n_assigned_unexposed: int
    n_assigned_exposed: int
    naive_gap: Estimate
    intention_to_treat: Estimate
    exposure_rate: Estimate
    complier_effect: Estimate
    complier_baseline: float


def analyse_exposure(
    frame: pd.DataFrame,
    outcome: str,
    confidence_level: float = 0.95,
) -> ExposureAnalysis:
    """Compare what exposure appears to show against what it identifies.

    ``exposure`` records whether the ad rendered, which is decided after
    assignment and only happens while the user is already browsing. Splitting
    the assigned arm on it therefore compares a selected subgroup against the
    whole control arm, and the gap that produces is not a treatment effect.

    The question behind that comparison is still answerable. Assignment is
    randomized and nobody in the control arm can be exposed, so assignment is a
    valid instrument for exposure under one-sided noncompliance, and the Wald
    ratio identifies the effect among users whose ad renders.
    """
    for column in ("treatment", "exposure", outcome):
        if column not in frame.columns:
            raise ValueError(f"Frame is missing the {column!r} column.")

    y = frame[outcome].to_numpy(dtype=float)
    z = frame["treatment"].to_numpy(dtype=float)
    d = frame["exposure"].to_numpy(dtype=float)
    for name, values in (("treatment", z), ("exposure", d), (outcome, y)):
        if not set(np.unique(values)).issubset({0.0, 1.0}):
            raise ValueError(f"{name} may contain only 0 and 1.")
    if d[z == 0].any():
        raise ValueError(
            "Some control users are marked exposed, so assignment is not a "
            "one-sided instrument and this estimator does not apply."
        )

    control = z == 0
    assigned = z == 1
    exposed = assigned & (d == 1)
    unexposed = assigned & (d == 0)

    rate_control = float(y[control].mean())
    rate_exposed = float(y[exposed].mean())
    rate_unexposed = float(y[unexposed].mean())

    naive_gap = Estimate.from_se(
        rate_exposed - rate_control,
        _two_proportion_se(y[exposed], y[control]),
        confidence_level,
    )
    intention_to_treat = Estimate.from_se(
        float(y[assigned].mean()) - rate_control,
        _two_proportion_se(y[assigned], y[control]),
        confidence_level,
    )
    exposure_rate = Estimate.from_se(
        float(d[assigned].mean()),
        float(d[assigned].std(ddof=1) / np.sqrt(int(assigned.sum()))),
        confidence_level,
    )

    wald, wald_se = _wald_ratio(y, z, d)
    complier_effect = Estimate.from_se(wald, wald_se, confidence_level)

    # Control mixes users who would have been exposed with users who would not.
    # The second group behaves the same in both arms under the exclusion
    # restriction, so its rate can be removed to recover what the exposed group
    # would have done untreated.
    share = exposure_rate.value
    complier_baseline = (rate_control - (1.0 - share) * rate_unexposed) / share

    return ExposureAnalysis(
        outcome=outcome,
        n=len(frame),
        rate_control=rate_control,
        rate_assigned_unexposed=rate_unexposed,
        rate_assigned_exposed=rate_exposed,
        n_control=int(control.sum()),
        n_assigned_unexposed=int(unexposed.sum()),
        n_assigned_exposed=int(exposed.sum()),
        naive_gap=naive_gap,
        intention_to_treat=intention_to_treat,
        exposure_rate=exposure_rate,
        complier_effect=complier_effect,
        complier_baseline=float(complier_baseline),
    )


def _wald_ratio(
    y: np.ndarray,
    z: np.ndarray,
    d: np.ndarray,
) -> tuple[float, float]:
    """Instrumental-variable estimate and its influence-function standard error.

    The ratio of covariances is the Wald estimator; its standard error comes
    from the influence function rather than from treating numerator and
    denominator as independent, because both are computed on the same users.
    """
    z_centered = z - z.mean()
    cov_dz = float((z_centered * d).mean())
    if abs(cov_dz) < 1e-12:
        raise ValueError("Assignment does not move exposure; the ratio is undefined.")
    beta = float((z_centered * y).mean() / cov_dz)
    influence = z_centered * (y - beta * d) / cov_dz
    return beta, float(influence.std(ddof=1) / np.sqrt(len(y)))


def _two_proportion_se(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.sqrt(
            left.var(ddof=1) / len(left) + right.var(ddof=1) / len(right)
        )
    )
