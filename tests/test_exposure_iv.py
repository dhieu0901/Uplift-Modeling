import numpy as np
import pandas as pd
import pytest

from src.evaluation.exposure_iv import analyse_exposure


def _one_sided_frame(
    n_control: int,
    n_unexposed: int,
    n_exposed: int,
    rate_control: float,
    rate_unexposed: float,
    rate_exposed: float,
) -> pd.DataFrame:
    def block(size: int, treatment: int, exposure: int, rate: float) -> pd.DataFrame:
        positives = int(round(rate * size))
        return pd.DataFrame(
            {
                "treatment": treatment,
                "exposure": exposure,
                "visit": [1] * positives + [0] * (size - positives),
            }
        )

    return pd.concat(
        [
            block(n_control, 0, 0, rate_control),
            block(n_unexposed, 1, 0, rate_unexposed),
            block(n_exposed, 1, 1, rate_exposed),
        ],
        ignore_index=True,
    )


def test_wald_ratio_matches_the_effect_it_is_built_from():
    # Never-exposed users behave identically in both arms, so the control rate
    # is a known mixture and the complier effect can be worked out by hand.
    frame = _one_sided_frame(
        n_control=10_000,
        n_unexposed=9_000,
        n_exposed=1_000,
        rate_control=0.10,
        rate_unexposed=0.05,
        rate_exposed=0.60,
    )

    result = analyse_exposure(frame, "visit")

    # Control = 0.9 * 0.05 + 0.1 * baseline  ->  baseline = 0.55
    assert result.complier_baseline == pytest.approx(0.55, abs=1e-9)
    assert result.complier_effect.value == pytest.approx(0.60 - 0.55, abs=1e-9)
    # And the naive reading overstates it by the amount of the selection.
    assert result.naive_gap.value == pytest.approx(0.50, abs=1e-9)


def test_the_apparent_gap_splits_into_effect_plus_selection():
    frame = _one_sided_frame(
        n_control=20_000,
        n_unexposed=19_000,
        n_exposed=1_000,
        rate_control=0.08,
        rate_unexposed=0.04,
        rate_exposed=0.50,
    )

    result = analyse_exposure(frame, "visit")
    selection = result.naive_gap.value - result.complier_effect.value

    assert selection > 0
    assert result.complier_baseline > result.rate_control
    np.testing.assert_allclose(
        result.naive_gap.value,
        result.complier_effect.value + selection,
    )


def test_an_exposed_control_user_makes_the_instrument_invalid():
    frame = _one_sided_frame(10, 10, 10, 0.1, 0.1, 0.5)
    frame.loc[0, "exposure"] = 1

    with pytest.raises(ValueError, match="one-sided"):
        analyse_exposure(frame, "visit")


def test_a_missing_column_is_refused():
    frame = _one_sided_frame(10, 10, 10, 0.1, 0.1, 0.5).drop(columns=["exposure"])

    with pytest.raises(ValueError, match="exposure"):
        analyse_exposure(frame, "visit")
