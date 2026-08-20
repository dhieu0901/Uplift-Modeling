from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_POLICY_COLORS = {
    "response_model": "#d97706",
    "random_targeting": "#94a3b8",
}
POLICY_COLOR = "#147d64"

#: One hue per base learner family, in the order the comparison reports them.
#: Checked for colorblind separation against a light surface rather than picked
#: by eye, so the three families stay distinguishable in print and for readers
#: with protanopia or deuteranopia.
BASE_FAMILY_COLORS = {
    "gradient_boosting": "#0e9f6e",
    "linear": "#d97706",
    "forest": "#3b5bdb",
}


def plot_policy_value_curve(
    table: pd.DataFrame,
    output_path: str | Path,
    title: str = "Locked-test AIPW incremental outcomes",
) -> Path:
    """Plot incremental outcomes with confidence intervals against budget.

    Matplotlib is imported lazily and forced onto a headless backend so that
    importing this module stays cheap for the test suite and for CI.
    """
    required = {"policy", "budget_pct", "incremental_outcome", "ci_lower", "ci_upper"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Policy table is missing required columns: {missing}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for policy, group in table.groupby("policy", sort=False):
        group = group.sort_values("budget_pct")
        center = group["incremental_outcome"].to_numpy(dtype=float)
        lower = group["ci_lower"].to_numpy(dtype=float)
        upper = group["ci_upper"].to_numpy(dtype=float)
        name = str(policy)
        axis.errorbar(
            group["budget_pct"].to_numpy(dtype=float),
            center,
            yerr=np.vstack([center - lower, upper - center]),
            marker="o",
            capsize=4,
            linewidth=2,
            color=REFERENCE_POLICY_COLORS.get(name, POLICY_COLOR),
            label=name,
        )
    axis.axhline(0.0, color="#5b6470", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("Targeting budget (%)")
    axis.set_ylabel("Incremental outcomes vs. no treatment")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_base_learner_comparison(
    table: pd.DataFrame,
    output_path: str | Path,
    title: str = "Selection-stage contrast vs response targeting, by base learner",
) -> Path:
    """Plot each candidate's selection contrast once per base learner family.

    Intervals rather than bars, because the selection rule reads a lower bound
    and a bar drawn to its tip would hide how wide the interval under it is.
    Candidates are ordered by their boosted-tree bound, which is the ordering
    the locked run produced, so a family that reorders them is visible as
    crossing rather than as a table the reader has to diff.
    """
    required = {"policy", "base_family", "difference", "ci_lower", "ci_upper"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Comparison table is missing required columns: {missing}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    present = {str(name) for name in table["base_family"]}
    families = [name for name in BASE_FAMILY_COLORS if name in present]
    families += sorted(present - set(families))
    reference = table[table["base_family"] == families[0]]
    ordered_policies = [
        str(policy)
        for policy in reference.sort_values("ci_lower", ascending=True)["policy"]
    ] or sorted({str(policy) for policy in table["policy"]})

    figure, axis = plt.subplots(figsize=(9.0, 0.9 * len(ordered_policies) + 2.0))
    offsets = np.linspace(0.26, -0.26, len(families))
    for family, offset in zip(families, offsets, strict=True):
        group = table[table["base_family"] == family]
        by_policy = dict(
            zip(
                [str(policy) for policy in group["policy"]],
                zip(
                    group["difference"].to_numpy(dtype=float),
                    group["ci_lower"].to_numpy(dtype=float),
                    group["ci_upper"].to_numpy(dtype=float),
                    strict=True,
                ),
                strict=True,
            )
        )
        positions, centers, lower, upper = [], [], [], []
        for index, policy in enumerate(ordered_policies):
            if policy not in by_policy:
                continue
            center, low, high = by_policy[policy]
            positions.append(index + offset)
            centers.append(center)
            lower.append(low)
            upper.append(high)
        center_array = np.asarray(centers, dtype=float)
        axis.errorbar(
            center_array,
            positions,
            xerr=np.vstack(
                [center_array - np.asarray(lower), np.asarray(upper) - center_array]
            ),
            marker="o",
            markersize=8,
            linestyle="none",
            linewidth=2,
            capsize=4,
            color=BASE_FAMILY_COLORS.get(family, POLICY_COLOR),
            label=family,
        )

    axis.axvline(0.0, color="#5b6470", linewidth=1)
    axis.set_yticks(range(len(ordered_policies)))
    axis.set_yticklabels(ordered_policies)
    axis.set_ylim(-0.6, len(ordered_policies) - 0.4)
    axis.set_title(title)
    axis.set_xlabel("Incremental outcomes vs response targeting")
    axis.grid(alpha=0.2, axis="x")
    # Below the axes rather than inside them. The intervals span the full width
    # of the plot, so any in-axes corner would sit on top of a result.
    axis.legend(
        title="Base learner",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(families),
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def dataframe_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Export a DataFrame as a Markdown table without requiring tabulate."""
    table = df.reset_index() if index else df.reset_index(drop=True)
    # Column labels are not always strings: a reset index or a numeric header
    # arrives as an int, and the widths below are measured in characters.
    columns = table.columns.astype(str).tolist()

    def format_value(value) -> str:
        if isinstance(value, (int, np.integer)):
            return str(value)
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "nan"
            return f"{value:.6f}"
        return str(value)

    rows = [
        [format_value(value).replace("|", "\\|") for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rows))
        if rows
        else len(columns[index])
        for index in range(len(columns))
    ]

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    header = render_row(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [render_row(row) for row in rows]
    return "\n".join([header, separator, *body])
