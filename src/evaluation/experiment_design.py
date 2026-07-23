from __future__ import annotations

from math import ceil
from statistics import NormalDist

import numpy as np


def policy_outcome_rate(
    no_campaign_rate: float,
    incremental_outcome: float,
    population_size: int,
) -> float:
    """Convert incremental outcomes into the outcome rate for a full policy arm."""
    no_campaign_rate = float(no_campaign_rate)
    incremental_outcome = float(incremental_outcome)
    if not 0.0 <= no_campaign_rate <= 1.0:
        raise ValueError("no_campaign_rate must be in the interval [0, 1].")
    if population_size <= 0:
        raise ValueError("population_size must be greater than zero.")
    result = no_campaign_rate + incremental_outcome / population_size
    if not 0.0 <= result <= 1.0:
        raise ValueError("The implied policy outcome rate is outside [0, 1].")
    return result


def two_proportion_sample_size(
    rate_a: float,
    rate_b: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """Calculate per-arm sample size for a test of two independent proportions."""
    rate_a = float(rate_a)
    rate_b = float(rate_b)
    if not 0.0 < rate_a < 1.0 or not 0.0 < rate_b < 1.0:
        raise ValueError("rate_a and rate_b must be in the interval (0, 1).")
    if rate_a == rate_b:
        raise ValueError("Sample size cannot be calculated when the effect size is zero.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in the interval (0, 1).")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in the interval (0, 1).")

    alpha_tail = alpha / 2.0 if two_sided else alpha
    z_alpha = NormalDist().inv_cdf(1.0 - alpha_tail)
    z_power = NormalDist().inv_cdf(power)
    pooled_rate = (rate_a + rate_b) / 2.0
    numerator = (
        z_alpha * np.sqrt(2.0 * pooled_rate * (1.0 - pooled_rate))
        + z_power
        * np.sqrt(
            rate_a * (1.0 - rate_a) + rate_b * (1.0 - rate_b)
        )
    ) ** 2
    return ceil(numerator / (rate_a - rate_b) ** 2)


def buffered_sample_size(sample_size: int, buffer_fraction: float = 0.15) -> int:
    """Add a buffer for attrition, logging loss, and assumption error."""
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero.")
    if buffer_fraction < 0.0:
        raise ValueError("buffer_fraction must not be negative.")
    return ceil(sample_size * (1.0 + buffer_fraction))
