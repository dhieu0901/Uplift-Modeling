from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_POLICY_COLORS = {
    "response_model": "#d97706",
    "random_targeting": "#94a3b8",
}
POLICY_COLOR = "#147d64"


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
        axis.errorbar(
            group["budget_pct"],
            center,
            yerr=np.vstack([center - lower, upper - center]),
            marker="o",
            capsize=4,
            linewidth=2,
            color=REFERENCE_POLICY_COLORS.get(policy, POLICY_COLOR),
            label=policy,
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


def dataframe_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Export a DataFrame as a Markdown table without requiring tabulate."""
    table = df.reset_index() if index else df.reset_index(drop=True)
    columns = [str(column) for column in table.columns]

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
