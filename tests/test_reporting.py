from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.reporting import (
    dataframe_to_markdown,
    plot_base_learner_comparison,
    plot_policy_value_curve,
)


def test_markdown_export_formats_numbers_and_escapes_pipes():
    table = pd.DataFrame(
        [
            {"policy": "a|b", "count": 3, "value": 1.5},
            {"policy": "c", "count": 40, "value": float("nan")},
        ]
    )

    rendered = dataframe_to_markdown(table).splitlines()

    assert rendered[0].startswith("| policy")
    assert set(rendered[1]) <= {"|", "-", " "}
    # Integers stay exact, floats get six decimals, and pipes are escaped so a
    # value can never break out of its cell.
    assert "a\\|b" in rendered[2]
    assert "1.500000" in rendered[2]
    assert "40" in rendered[3]
    assert "nan" in rendered[3]
    assert len(rendered) == 4


def test_markdown_export_handles_an_empty_frame():
    rendered = dataframe_to_markdown(pd.DataFrame(columns=["policy", "value"]))

    assert rendered.splitlines()[0].startswith("| policy")
    assert len(rendered.splitlines()) == 2


def test_markdown_export_can_include_the_index():
    table = pd.DataFrame({"value": [1, 2]}, index=["first", "second"])

    rendered = dataframe_to_markdown(table, index=True)

    assert "first" in rendered
    assert "second" in rendered


def test_policy_value_plot_writes_a_figure(tmp_path: Path):
    table = pd.DataFrame(
        [
            {
                "policy": policy,
                "budget_pct": budget,
                "incremental_outcome": 10.0 * budget,
                "ci_lower": 10.0 * budget - 5.0,
                "ci_upper": 10.0 * budget + 5.0,
            }
            for policy in ("response_model", "random_targeting", "s_learner")
            for budget in (5.0, 10.0, 20.0)
        ]
    )
    output_path = tmp_path / "nested" / "policy_value.png"

    written = plot_policy_value_curve(table, output_path)

    assert written == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_policy_value_plot_rejects_an_incomplete_table(tmp_path: Path):
    table = pd.DataFrame([{"policy": "a", "budget_pct": 5.0}])

    with pytest.raises(ValueError, match="missing required columns"):
        plot_policy_value_curve(table, tmp_path / "figure.png")


def _comparison_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "base_family": family,
                "policy": policy,
                "difference": difference,
                "ci_lower": difference - 40.0,
                "ci_upper": difference + 40.0,
            }
            for family, offsets in (
                ("gradient_boosting", (90.0, 30.0, -20.0)),
                ("linear", (10.0, 70.0, -5.0)),
                ("forest", (55.0, 15.0, 25.0)),
            )
            for policy, difference in zip(
                ("s_learner", "t_learner", "cvt"), offsets, strict=True
            )
        ]
    )


def test_base_learner_plot_writes_a_figure(tmp_path: Path):
    output_path = tmp_path / "nested" / "base_learners.png"

    written = plot_base_learner_comparison(_comparison_table(), output_path)

    assert written == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_base_learner_plot_accepts_a_subset_of_families(tmp_path: Path):
    """A comparison of two families has to draw, not fail on the missing third."""
    table = _comparison_table()
    table = table[table["base_family"] != "forest"]

    written = plot_base_learner_comparison(table, tmp_path / "two.png")

    assert written.exists()


def test_base_learner_plot_rejects_an_incomplete_table(tmp_path: Path):
    table = pd.DataFrame([{"policy": "s_learner", "base_family": "linear"}])

    with pytest.raises(ValueError, match="missing required columns"):
        plot_base_learner_comparison(table, tmp_path / "figure.png")
